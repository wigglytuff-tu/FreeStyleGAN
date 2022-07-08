import argparse
import numpy as np
import os, sys
import warnings
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

def upDir(x):
   return os.path.dirname(x)

def getDir():
    return upDir(os.path.realpath(__file__))

sys.path.append(upDir(upDir(getDir())))
sys.path.append(upDir(upDir(upDir(getDir()))))

import dnnlib
import dnnlib.tflib as tflib
import tensorflow as tf

import pipeline
from graphics_utils.image_io import *
from graphics_utils.tf_io import *
import multiview.multiview_prep
import training_utils
           
#----------------------------------------------------------------------------

# create fixed noise maps for the generator
def createNoiseInstance():
    resList = [ 2**(i+2) for i in range(9) for _ in (0,1) ][1:]
    noises = []
    for res in resList:
        noises.append(np.random.normal(size=(1,1,res,res)).astype(np.float32))
    return noises
    
#----------------------------------------------------------------------------

# train stage 1 of implicit latent representation network
def train(data_dir, log_dir, **_kwargs):

    data_dir = os.path.join(data_dir, "freestylegan", "")
    general_data_dir = os.path.join(upDir(upDir(getDir())), "data", "")

    #---------------------------------------

    np.random.seed(100)
    
    mlpData = dict()
    
    # architecture params
    mlpData["layerCount"] = 3
    mlpData["nonlinearity"] = 'lrelu'
    mlpData["fmaps"] = 32
    mlpData["modStyleCount"] = 6            # number of styles considered variable
    mlpData["pretrained"] = False
    mlpData["meanLatentSamples"] = 1000     # number of samples to determine latent mean

    # training params
    mlpData["batchSize"] = 2
    mlpData["iterations"] = 7500 
    mlpData["learningRate"] = 0.005
    mlpData["decaySteps"] = 200
    mlpData["decayRate"] = 0.98
    mlpData["noiseStrength"] = 0.1          # amount of noise applied to latents during training
    mlpData["noiseRampLength"] = .8         # decay schedule of latent noise    
    
    # loss params
    mlpData["vggWeight"] = 250.
    mlpData["vggResolution"] = 256
    mlpData["vggFaceWeight"] = 5.
    mlpData["priorWeightFlip"] = 1./3.      # when to switch the prior weight during training
    mlpData["priorWeightPhase1"] = 10.
    mlpData["priorWeightPhase2"] = 0.1

    # logging params
    snapshotInterval = 50                   # how often to output images during training
    saveSnapshotsInTB = True                # where to store the images (tensorboard or regular files?)

    #---------------------------------------

    # load cameras and aligned images
    print("Loading input views...")
    origCams, alignedImages = multiview.multiview_prep.loadViews(
        data_dir, 
        loadAlignedImages=True, 
        normalizeImages=False,
        appendAlpha=True,
        upload=False)
    inputManiCoords, inputImages, _ = multiview.multiview_prep.filterValidViews(
        origCams, alignedImages.tex, general_data_dir)
    assert len(inputManiCoords) > 0, "No valid images."
    inputVggFaceFeats = training_utils.loadInputVGGFaceFeats(data_dir)

    # build/load/combine networks
    print('Initializing Tensorflow...')
    tflib.init_tf()
    sess = tf.get_default_session()

    print('Loading pretrained generator...')
    gan_file = os.path.join(general_data_dir + "networks", "stylegan2_generator.pkl")
    gan = pickle.load(open(gan_file, "rb"))

    print("Computing latent mean...")    
    M = gan.components.mapping
    z = np.random.normal(size=(mlpData["meanLatentSamples"], *M.input_shape[1:]))
    w = M.run(z, None)[:, 0, :]
    latentMean = np.mean(w, axis=0)
    
    # the MLP predicts the offset from the latent mean
    mlpData["dynamicLatentsBase"] = np.tile(latentMean[None, ::], (1, mlpData["modStyleCount"], 1))

    # the static latents are initialized with the latent mean
    staticLatentsInit = np.tile(latentMean[None, ::], (1, 18-mlpData["modStyleCount"], 1))

    # manually set the (fixed) noise maps
    mlpData["noises"] = createNoiseInstance()
    
    # strength of latent perturbations during training
    noiseStrength = tf.placeholder(tf.float32, shape=(), name="noiseStrength")

    # put the pipeline together
    nodes = pipeline.build(
        gan=gan,
        mlp=mlpData,
        staticLatents=staticLatentsInit, 
        batchSize=mlpData["batchSize"],
        noiseStrength=noiseStrength)

    # loss setup 
    print("Setting up loss and optimizer...")
    
    nodes.gtImage = tf.placeholder(tf.float32, shape=(None, 1024, 1024, 4), name="gtImage")

    with tf.variable_scope('Loss'):
        gtImg = tflib.convert_images_from_uint8(nodes.gtImage[..., 0:3], nhwc_to_nchw=True)
        
        # L1
        training_utils.l1LossSetup(nodes, gtImg)

        # VGG
        training_utils.vggLossSetup(mlpData, nodes, gtImg)
        
        # Identity
        training_utils.identityLossSetup(mlpData, nodes, general_data_dir, inputVggFaceFeats)
        
        # Prior
        nodes.priorWeight = tf.placeholder(tf.float32, shape=(), name="priorWeight")
        training_utils.priorLossSetup(mlpData, nodes, nodes.priorWeight)

        # combine losses
        nodes.loss = nodes.l1Loss + nodes.vggLoss + nodes.vggFaceLoss + nodes.priorLoss

    # optimizer setup
    global_step = tf.Variable(0, trainable=False)
    adaptiveLR = tf.train.exponential_decay(mlpData["learningRate"], global_step, mlpData["decaySteps"], mlpData["decayRate"], staircase=True)
    latentVars = [var for var in tf.trainable_variables() if 'staticLatents' in var.name]
    mlpVars = [var for var in tf.trainable_variables() if 'ImplicitLatent' in var.name]
    trainableVars = latentVars + mlpVars

    optimizer = tf.train.AdamOptimizer(adaptiveLR)
    trainStep = optimizer.minimize(nodes.loss, global_step=global_step, var_list=trainableVars)

    # prepare logging
    writer = tf.summary.FileWriter(logdir=log_dir)
    writer.add_graph(sess.graph)
    writer.flush()

    tflib.init_uninitialized_vars()

    #-----------------------------
    def saveNetwork(iter=0):
        modelSavePath = os.path.join(data_dir, "model_stage1_")
        mlpData["pretrained"] = True
        mlpData["staticLatents"] = sess.run(nodes.staticLatents)
        for var in mlpVars:
            mlpData[var.name] = sess.run(var)
        pickle.dump(mlpData, open(modelSavePath + str(iter) + ".pickle", "wb") )
    #-----------------------------

    # training loop
    print("Starting training...")
    
    feedDict = {}

    for it in range(mlpData["iterations"]):
    
        # collect iteration data
        frac = it / mlpData["iterations"]
        lr = sess.run(adaptiveLR)
        priorWeight = mlpData["priorWeightPhase1"] if frac < mlpData["priorWeightFlip"] else mlpData["priorWeightPhase2"]
        noiseStrength = mlpData["noiseStrength"] * max(0.0, 1.0 - frac / mlpData["noiseRampLength"]) ** 2        
        print("\rit=%i/%i | lr=%.4f | prior=%.1f | n=%.2f" % \
                (it, mlpData["iterations"], lr, priorWeight, noiseStrength), end='')

        # create a batch
        indices = np.random.choice(len(inputImages), mlpData["batchSize"])
        trainImages = np.stack([inputImages[i] for i in indices], axis=0)
        trainCoords = np.stack([inputManiCoords[i] for i in indices], axis=0)

        # SGD step
        feedDict[nodes.gtImage] = trainImages
        feedDict[nodes.maniCoord] = trainCoords
        feedDict[nodes.priorWeight] = priorWeight
        feedDict[nodes.noiseStrength] = noiseStrength
        _, loss, l1, vgg, prior, vggFace = sess.run(
            (trainStep, nodes.loss, nodes.l1Loss, nodes.vggLoss, nodes.priorLoss, nodes.vggFaceLoss), 
            feedDict)
            
        # logging
        loss_sum = tf.Summary.Value(tag="loss", simple_value=loss)
        l1_sum = tf.Summary.Value(tag="l1", simple_value=l1)
        vgg_sum = tf.Summary.Value(tag="vgg", simple_value=vgg)
        prior_sum = tf.Summary.Value(tag="prior", simple_value=prior)
        vggFace_sum = tf.Summary.Value(tag="vggface", simple_value=vggFace)
        noiseStrength_sum = tf.Summary.Value(tag="noiseStrength", simple_value=noiseStrength)
        writer.add_summary(tf.Summary(
            value=[loss_sum, l1_sum, vgg_sum, prior_sum, vggFace_sum, noiseStrength_sum]), 
            it)

        if it % snapshotInterval == 0 or it == mlpData["iterations"] - 1:
            outImg = sess.run(nodes.displayOutput, feedDict)[0]
            outComp = np.concatenate([trainImages[0, :, :, 0:3], outImg], axis=1) / 255.
            if saveSnapshotsInTB:
                writeImageToTensorboard(outComp, writer, it, res=(1024, 512), tag="images")
            else:
                saveImage(outComp, log_dir + "/it_" + str(it).zfill(5) + ".png")

        if it % 10 == 0:
            writer.flush()

    print("")

    print("Saving network...")
    saveNetwork()

    # export images
    print("Saving images...")
    imgDir = os.path.join(data_dir, "inputEmbeddings")
    if not os.path.exists(imgDir):
        os.mkdir(imgDir)
    for idx, (mc, img) in enumerate(zip(inputManiCoords, inputImages)):
        feedDict[nodes.maniCoord] = mc[None, ...]
        prediction = sess.run(nodes.displayOutput, feedDict)[0]
        comp = np.concatenate([img[..., 0:3], prediction], axis=1) / 255.   
        saveImage(comp, imgDir + "/emb_" + str(idx).zfill(2) + ".png")

    sess.close()
    
#----------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description='''Training mapping from camera manifold to latent code - stage 1.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data-dir', help='Dataset directory', required=True)
    parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='../../results', metavar='DIR')
    parser.add_argument('--log-dir', help='Root directory for logs (default: %(default)s)', default='../../logs', metavar='DIR')

    args = parser.parse_args()
    kwargs = vars(args)

    if not os.path.exists(kwargs['log_dir']):
        os.mkdir(kwargs['log_dir'])
    experimentID = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    kwargs['log_dir'] = os.path.join(kwargs['log_dir'], experimentID)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')

    dnnlib.submit_run(sc, 'training_stage1.train', **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":  
    main()
    print("=== TERMINATED ===")

#----------------------------------------------------------------------------
