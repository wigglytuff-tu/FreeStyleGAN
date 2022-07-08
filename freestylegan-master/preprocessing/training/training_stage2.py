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
from dnnlib import EasyDict
import tensorflow as tf

import pipeline
import cameras.camera_manifold as manifold

from graphics_utils.ogl_ops import *
from graphics_utils.render_utils import *
from graphics_utils.image_io import *
from graphics_utils.tf_io import *
import multiview.multiview_prep as mvprep
import training_utils

#----------------------------------------------------------------------------

# produce ULR rendering on the manifold
def renderSample(mesh, origImgs, origCams, renderData, maniCoords, mouth3D, resolution):
    
    # optimize camera
    manifoldCam = manifold.manifoldCoordToCamera(maniCoords, mouth3D)
    
    # render image
    gBuffer = renderData.gBufferOp.render(mesh, manifoldCam, resolution)
    rendering = renderData.ulrOp.render(
        gBuffer, manifoldCam.position,
        origImgs.tex, origImgs.gBuffers, origCams.camBuffer,
        varianceWeightToAlpha=True)
    
    # make variance estimate slightly more conservative
    rendering = renderData.morphologyOp.render(rendering, kernelSize=2, erode=True, alphaOnly=True)
    
    rendering = downloadImage(rendering)
    rendering[..., 0:3] *= 255.
    return rendering

#----------------------------------------------------------------------------

# train stage 2 of implicit latent representation network
def train(data_dir, log_dir, **_kwargs):
    data_dir = os.path.join(data_dir, "freestylegan", "")
    general_data_dir = os.path.join(upDir(upDir(getDir())), "data", "")

    #---------------------------------------

    np.random.seed(100)
    
    mlpFile = "model_stage1_0.pickle"
    mlpData =  pickle.load(open(data_dir + mlpFile, "rb"))

    # training params
    mlpData["batchSize"] = 2
    mlpData["iterations"] = 750
    mlpData["learningRate"] = 0.005
    mlpData["decaySteps"] = 100
    mlpData["decayRate"] = 0.9
    mlpData["origImageProb"] = .15          # probability of showing original images during training
    
    # loss params
    mlpData["vggFaceWeight"] = 5
    mlpData["priorWeight"] = 0.1

    # logging params
    snapshotInterval = 50                   # how often to output images during training
    saveSnapshotsInTB = True                # where to store the images (tensorboard or regular files?)

    #---------------------------------------

    resolution = 1024
    print("Setting up OpenGL...")
    oglInit()
    print("OpenGL initialized")
    renderData = EasyDict()
    fbo = createFramebuffer()
    renderData.gBufferOp = GBufferOP(fbo, resolution)
    renderData.ulrOp = ULROP(fbo, resolution)
    renderData.morphologyOp = MorphologyOP(fbo, resolution)

    # load geometry, unaligned images, and cameras
    print("Loading mesh, input views, and preparing renderings...")
    mesh = pickle.load(open(os.path.join(data_dir, "mesh.pickle"), "rb"))
    addBackgroundPlane(mesh, depth=5., res=1)
    uploadMesh(mesh, uploadTexture=False)
    origCams, origImgs = mvprep.loadViews(data_dir)
    origImgs.gBuffers = mvprep.renderInputViewGBuffers(renderData.gBufferOp, mesh, origImgs, origCams)
    mouthPos = np.loadtxt(data_dir + 'mouthPosition.txt')
    inputVggFaceFeats = training_utils.loadInputVGGFaceFeats(data_dir)

    # load aligned images
    _, alignedImages = mvprep.loadViews(
        data_dir, 
        loadAlignedImages=True, 
        normalizeImages=False,
        appendAlpha=True,
        upload=False)
    inputManiCoords, inputImages, manifoldClampData = mvprep.filterValidViews(
        origCams, alignedImages.tex, general_data_dir)
    assert len(inputManiCoords) > 0, "No valid images."

    # build/load/combine networks
    tflib.init_tf()
    sess = tf.get_default_session()
    print('Loading pretrained generator...')
    gan_file = os.path.join(general_data_dir, "networks", "stylegan2_generator.pkl")

    gan = pickle.load(open(gan_file, "rb"))

    nodes = pipeline.build(
        gan=gan,
        mlp=mlpData,
        staticLatents=mlpData["staticLatents"], 
        batchSize=mlpData["batchSize"])

    # loss setup 
    print("Setting up loss and optimizer...")
    
    nodes.gtImage = tf.placeholder(tf.float32, shape=(None, 1024, 1024, 4), name="gtImage")

    with tf.variable_scope('Loss'):
        gtImg = tflib.convert_images_from_uint8(nodes.gtImage[..., 0:3], nhwc_to_nchw=True)
        
        # masked L1
        mask = tf.transpose(nodes.gtImage[..., 3:], [0, 3, 1, 2])
        training_utils.l1LossSetup(nodes, gtImg, mask)
        
        # Prior
        training_utils.priorLossSetup(mlpData, nodes, mlpData["priorWeight"])
            
        # Identity
        training_utils.identityLossSetup(mlpData, nodes, general_data_dir, inputVggFaceFeats)
        
        # combine losses
        nodes.loss = nodes.l1Loss + nodes.priorLoss + nodes.vggFaceLoss

    # optimizer setup
    global_step = tf.Variable(0, trainable=False)
    adaptiveLR = tf.train.exponential_decay(mlpData["learningRate"], global_step, mlpData["decaySteps"], mlpData["decayRate"], staircase=True)
    mlpVars = [var for var in tf.trainable_variables() if 'ImplicitLatent' in var.name]

    optimizer = tf.train.AdamOptimizer(adaptiveLR)
    trainStep = optimizer.minimize(nodes.loss, global_step=global_step, var_list=mlpVars)

    writer = tf.summary.FileWriter(logdir=log_dir)
    writer.add_graph(sess.graph)
    writer.flush()

    tflib.init_uninitialized_vars()

    #-----------------------------
    def saveNetwork(iter=0):
        modelSavePath = os.path.join(data_dir, "model_stage2_")
        for var in mlpVars:
            mlpData[var.name] = sess.run(var)
        pickle.dump(mlpData, open(modelSavePath + str(iter) + ".pickle", "wb") )
    #-----------------------------

    # training loop
    print("Starting training...")

    trainImages = np.zeros((mlpData["batchSize"], resolution, resolution, 4), dtype=np.float32)
    trainCoords = np.zeros((mlpData["batchSize"], 3), dtype=np.float32)
    
    feedDict = {}

    for it in range(mlpData["iterations"]):
    
        # collect iteration data
        lr = sess.run(adaptiveLR)
        print("\rit=%i/%i | lr=%.4f" % (it, mlpData["iterations"], lr), end='')

        # create a batch
        for b in range(mlpData["batchSize"]):
            if np.random.random() < mlpData["origImageProb"]:
                # show input images with some probability
                index = np.random.choice(len(inputImages))
                trainCoords[b] = inputManiCoords[index]
                trainImages[b] = inputImages[index]
            else:
                # render ULR view otherwise
                trainCoords[b] = manifold.sampleManifold(manifoldClampData)
                trainImages[b] = renderSample(mesh, origImgs, origCams, renderData, trainCoords[b], mouthPos, resolution)

        # SGD step
        feedDict[nodes.gtImage] = trainImages
        feedDict[nodes.maniCoord] = trainCoords
        _, loss, l1, prior, vggFace = sess.run(
            (trainStep, nodes.loss, nodes.l1Loss, nodes.priorLoss, nodes.vggFaceLoss), 
            feedDict)
            
        # logging
        loss_sum = tf.Summary.Value(tag="loss", simple_value=loss)
        l1_sum = tf.Summary.Value(tag="l1", simple_value=l1)
        prior_sum = tf.Summary.Value(tag="prior", simple_value=prior)
        vggFace_sum = tf.Summary.Value(tag="vggface", simple_value=vggFace)
        writer.add_summary(tf.Summary(value=[loss_sum, l1_sum, prior_sum, vggFace_sum]), it)

        if it % snapshotInterval == 0 or it == mlpData["iterations"] - 1:
            outImg = sess.run(nodes.displayOutput, feedDict)[0]
            trainImg = trainImages[0, ...]
            trainImg = overCompositing(trainImg, np.full_like(trainImg, 255.))
            outComp = np.concatenate([trainImg, outImg], axis=1) / 255.
            if saveSnapshotsInTB:
                writeImageToTensorboard(outComp, writer, it, res=(1024, 512), tag="images")
            else:
                saveImage(outComp, log_dir + "/it_" + str(it).zfill(5) + ".png")

        if it % 10 == 0:
            writer.flush()

    print("")

    print("Saving network...")
    saveNetwork()

    sess.close()

#----------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description='''Training mapping from camera manifold to latent code - stage 2.

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

    dnnlib.submit_run(sc, 'training_stage2.train', **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":  
    main()
    print("=== TERMINATED ===")

#----------------------------------------------------------------------------
