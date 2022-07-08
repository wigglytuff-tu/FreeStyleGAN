import sys, os
import numpy as np
import tensorflow as tf
from training import misc

def upDir(x):
   return os.path.dirname(x)

def getPreprocessingDir():
    return upDir(os.path.realpath(__file__))

sys.path.append(upDir(getPreprocessingDir()))

import vgg_face

#----------------------------------------------------------------------------

# load VGG-face features and average them over all input views
def loadInputVGGFaceFeats(data_dir):
    inputVggFaceFeats = np.loadtxt(data_dir + "vggFaceFeats.txt").astype(np.float32)
    inputVggFaceFeats = np.mean(inputVggFaceFeats, axis=0)
    inputVggFaceFeats /= np.linalg.norm(inputVggFaceFeats)
    return inputVggFaceFeats

#----------------------------------------------------------------------------

def l1LossSetup(nodes, gtImg, mask=None):
    absError = tf.abs(nodes.imageOutput - gtImg)
    if mask is not None:
        absError *= mask
    nodes.l1Loss = tf.reduce_mean(absError)

#----------------------------------------------------------------------------

# set up LPIPS loss
def vggLossSetup(mlpData, nodes, gtImg):

    def downsampling(img, res=mlpData["vggResolution"]):
        sh = img.shape.as_list()
        factor = sh[2] // res
        return tf.reduce_mean(tf.reshape(img, [-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor]), axis=[3,5])
    
    lpips = misc.load_pkl('http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/vgg16_zhang_perceptual.pkl')
    downOutput = downsampling(nodes.imageOutput)
    downGtImg = downsampling(gtImg)
    vgg_dst = lpips.get_output_for(downOutput, downGtImg)
    nodes.vggLoss = mlpData["vggWeight"] * tf.reduce_mean(vgg_dst)

#----------------------------------------------------------------------------

# set up W+ prior loss
def priorLossSetup(mlpData, nodes, priorWeight):

    def meanDeviationLoss(x):
        xMean = tf.reduce_mean(x, axis=1, keepdims=True)
        return tf.reduce_mean(tf.abs(x - xMean))

    with tf.variable_scope('Prior'):
        staticLatents = nodes.cleanLatents[::, mlpData["modStyleCount"]:, ::]
        dynamicLatents = nodes.cleanLatents[::, :mlpData["modStyleCount"], ::]
        dynamicLatentsLoss = meanDeviationLoss(dynamicLatents)
        staticLatentsLoss = meanDeviationLoss(staticLatents)
        varyingFraction = mlpData["modStyleCount"] / 18.
        nodes.priorLoss = varyingFraction * dynamicLatentsLoss + (1-varyingFraction) * staticLatentsLoss
        nodes.priorLoss = priorWeight * nodes.priorLoss

#----------------------------------------------------------------------------

# set up VGGFace-based identity loss
def identityLossSetup(mlpData, nodes, general_data_dir, inputVggFaceFeats):

    with tf.variable_scope("VGG-Face"):
        vggFaceFeats = vgg_face.vgg_face_features(
            general_data_dir + 'networks/vgg-face.mat', 
            nodes.oglOutput * 255)
        vggDst =  1. - tf.einsum('ij,j->i', vggFaceFeats, tf.constant(inputVggFaceFeats))
        nodes.vggFaceLoss = mlpData["vggFaceWeight"] * tf.reduce_mean(vggDst)