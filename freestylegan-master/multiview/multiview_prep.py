import os, sys
import numpy as np
import pickle
import cv2

def getMVDir():
    return os.path.dirname(os.path.realpath(__file__))

sys.path.append(getMVDir())
sys.path.append(os.path.dirname(getMVDir()))
from dnnlib.util import EasyDict

from graphics_utils.image_io import loadImage
from graphics_utils.ogl_ops import *
import cameras.camera_manifold as manifold

#=======================================================
# load input images and cameras
def loadViews(
    data_dir, 
    loadAlignedImages=False, 
    normalizeImages=True,
    appendAlpha=False,
    upload=True,
    resizeFactor=1,
    leaveOut=-1,
    blacklistSuffix=""):

    # check for blacklist
    blacklistFile = data_dir + 'blacklist_' + blacklistSuffix + '.txt'

    blacklist = []
    if os.path.isfile(blacklistFile):
        blacklist = np.loadtxt(blacklistFile, dtype=int, ndmin=1)
    else:
        assert blacklistSuffix == "", "Could not find blacklist."

    if leaveOut != -1 and leaveOut not in blacklist:
        blacklist.append(leaveOut)

    if len(blacklist) > 0:
        print(" Ignoring image(s):", *blacklist)
    
    # cameras
    cams = EasyDict()
    cams.cams = pickle.load(open(data_dir + 'cams_aligned.pickle', "rb"))
    cams.manifoldCoords = []

    cams.modelMatrix = np.loadtxt(data_dir + 'modelMatrix.txt')

    # images & G-buffers
    imgs = EasyDict()
    imgs.tex = []
    imgs.res = []
    imgs.transposed = []

    for i in range(len(cams.cams)):
        
        if i in blacklist:
            continue
        
        imgFile = data_dir + "img_" + str(i).zfill(3) + "_"
        imgFile += "aligned.png" if loadAlignedImages else "orig.png"
        
        img = loadImage(imgFile, normalize=normalizeImages, appendAlpha=appendAlpha)
        if resizeFactor != 1:
            res = img.shape[0:2][::-1]
            newRes = (int(res[0] * resizeFactor), int(res[1] * resizeFactor))
            img = cv2.resize(img, newRes, interpolation=cv2.INTER_AREA)
        imgs.res.append(img.shape[0:2])

        if img.shape[0] < img.shape[1]:
            imgs.transposed.append(True)
            img = np.transpose(img, [1, 0, 2])       
        else:
            imgs.transposed.append(False)
        
        imgs.tex.append(img)

        manifoldCoords, _ = manifold.projectToManifoldCam(cams.cams[i])
        cams.manifoldCoords.append(manifoldCoords)

    # clean up camera list
    if len(blacklist) != 0:
        tmpCams = []
        for i, c in enumerate(cams.cams):
            if i not in blacklist:
                tmpCams.append(c)
        cams.cams = tmpCams

    imgs.imgCount = len(cams.cams)
    cams.manifoldCoords = np.array(cams.manifoldCoords)

    # upload images and cameras
    if upload:
        imgs.tex = uploadImageList(imgs.tex)
        cams.camBuffer = uploadCameras(cams.cams, imgs.transposed)

    return cams, imgs

#=======================================================

# retain only images that correspond to valid manifold coordinates
def filterValidViews(cams, images, general_data_dir):
    manifoldClampData = np.loadtxt(general_data_dir + "manifoldClamp.txt")
    filteredManifoldCoords = []
    filteredImages = []
    counter = 0
    for mc, img in zip(cams.manifoldCoords, images):
        clamped = manifold.clampCoordinateToManifold(mc, manifoldClampData)
        if np.array_equal(mc, clamped):
            filteredManifoldCoords.append(mc)
            filteredImages.append(img)
            counter += 1
    print("%i of %i input views are valid" % (counter, len(images)))
    return filteredManifoldCoords, filteredImages, manifoldClampData

#=======================================================

# render depth buffer for all input views
def renderInputViewGBuffers(gBufferOp, mesh, inputImgs, inputCams):
    inputGBuffers = []
    for i in range(len(inputCams.cams)):
        imgRes = inputImgs.res[i]
        gBufferTex = gBufferOp.render(mesh, inputCams.cams[i], imgRes, mode='Depth')
        gBuffer = downloadImage(gBufferTex)
        if imgRes[0] < imgRes[1]:
            gBuffer = np.transpose(gBuffer, [1, 0, 2])
        inputGBuffers.append(gBuffer)
    return uploadImageList(inputGBuffers)