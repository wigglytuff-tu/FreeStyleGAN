import sys, os
import numpy as np
import pickle
import scipy.ndimage
import matplotlib.pyplot as plt
from scipy.optimize import root
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf

def upDir(x):
   return os.path.dirname(x)

def getAlignmentDir():
    return upDir(os.path.realpath(__file__))

sys.path.append(upDir(getAlignmentDir()))
sys.path.append(upDir(upDir(getAlignmentDir())))

import landmarks
import vgg_face
from cameras.camera import *
import cameras.alignment_coords
import cameras.camera_manifold as manifold
from graphics_utils.ogl_ops import *
from graphics_utils.image_io import *
from graphics_utils.render_utils import *
import multiview.mvs_io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#=======================================================

# find 2D similarity transform aligning 2D facial features 
def findImageAlignment(p0):

    A = np.zeros([4, 4])
    rhs = np.zeros(4)

    p1 = cameras.alignment_coords.getReferencePixelAlignmentFeats()

    for i in range(2):
        y, x = p0[i]
        yp, xp = p1[i]
        A[2 * i, 0    ] = x
        A[2 * i, 1    ] = -y
        A[2 * i, 2    ] = 1.
        A[2 * i + 1, 0] = y
        A[2 * i + 1, 1] = x
        A[2 * i + 1, 3] = 1.
        rhs[2 * i    ] = xp
        rhs[2 * i + 1] = yp

    solVec = np.linalg.solve(A, rhs)
    solMat = np.array(
        [[solVec[0], -solVec[1],  0, solVec[2]], 
        [ solVec[1],  solVec[0],  0, solVec[3]], 
        [         0,          0, 1.,         0], 
        [         0,          0,  0,        1.]] )
    return np.linalg.inv(solMat)
    
#=======================================================

# find 3D similarity transform aligning 3D facial features to canonical coordinates
def findMeshAlignment(featurePos3D):

    leftEyePosition, rightEyePosition, mouthPosition = featurePos3D
    midEyePosition = (leftEyePosition + rightEyePosition) / 2
    leftEyePosition = np.hstack([leftEyePosition, 1]) 
    rightEyePosition = np.hstack([rightEyePosition, 1])
    mouthPosition = np.hstack([mouthPosition, 1])

    T = getTranslationMatrix(-midEyePosition)

    #-------------------------------------------------
    def fullTransform(alpha, beta, gamma, scale):
        R = getRotationMatrix([alpha, beta, gamma])
        S = getIsotropicScalingMatrix(scale)
        return R.dot(S).dot(T)
    #-------------------------------------------------

    def objective(params):
        alpha, beta, gamma, scale = params
        M = fullTransform(alpha, beta, gamma, scale)
        l = M.dot(leftEyePosition)
        r = M.dot(rightEyePosition)
        c = M.dot(mouthPosition)

        eyes3D = cameras.alignment_coords.eyes3D
        mouthZ = cameras.alignment_coords.mouthZ

        f1 = (l - eyes3D[0])[0:3]   # left eye should be at (-1, 0, 0)
        f2 = (r - eyes3D[1])[0:3]   # right eye shoult be at (1, 0, 0)
        f3 = c[2] - mouthZ          # mouth should be at (?, ?, 0.25)
        return [*f1, *f2, f3]

    result = root(objective, [0, 0, 0, 1], method='lm')
    return fullTransform(*result.x)
    
#=======================================================

# run face alignment
def main(argv):
    path = argv[1]
    useBlurred = int(argv[2]) == 1
    outres = 1024
    
    dataDir = upDir(upDir(getAlignmentDir())) + "/data/"

    #-------------------------------------
    print("Loading scene...")
    dir_list = os.listdir(path)
    if "realitycapture" in dir_list:
        print("Run data from realitycapture")
        mesh, cameras, images = multiview.mvs_io.importRealityCaptureData(path, useBlurred)
    elif "colmap" in dir_list:
        print("Run data from colmap")
        mesh, cameras, images = multiview.mvs_io.importColmapData(path, useBlurred)
    else:
        assert False, "directory name cannot be matched!"

    #-------------------------------------
    print("Detecting facial features...")
    featureCoords = []

    detector, predictor = landmarks.initDetectors()

    for idx, img in enumerate(images):
        img = (img[..., 0:3] * 255).astype(np.uint8)
        success, lm = landmarks.detectLandmarks(img, detector, predictor)
        
        # automatic feature detection succeeded
        if success:
            print("Image %i: Facial feature detection successful." % idx)
            feats = landmarks.findFeatures(lm)
        
        # automatic feature detection failed, resort to manual annotation
        else:
            print("Image %i: No facial features found. Manual annotation needed." % idx)
            
            fig, ax = plt.subplots()
            ax.imshow(img, interpolation='nearest')

            feats = []

            def onclick(event):
                feats.append(np.array([event.xdata, event.ydata]))

            def onkey(event):
                plt.close()
            
            fig.canvas.mpl_connect('button_press_event', onclick)
            fig.canvas.mpl_connect('key_press_event', onkey)

            fig.suptitle("Click on the following features in order: 1. Left eye. 2. Right eye. 3. Mouth. Press any key to confirm.")
                
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()

            if len(feats) != 3:
                print("Need three features. Ignoring this image...")
                continue
        
        featureCoords.append(feats)

    #-------------------------------------
    print("Reprojecting features...")

    outDir = os.path.join(path, "freestylegan", "")
    if not os.path.exists(outDir):
        os.mkdir(outDir) 

    resolution = images[0].shape[0:2]
    
    # initialize OpenGL
    oglInit()
    uploadMesh(mesh, uploadTexture=False)
    fbo = createFramebuffer()
    gBufferOp = GBufferOP(fbo, resolution)

    # estimate eye world space positions by averaging reprojections of feature coords
    featureEstimates = np.zeros((3, 4), dtype=np.float32)
    for features, image, camera in zip(featureCoords, images, cameras):
        
        if not features:
            continue

        # render position buffer
        positionBufferTex = gBufferOp.render(mesh, camera, image.shape[0:2][::-1])
        positionBuffer = downloadImage(positionBufferTex)

        # round features to integers
        featuresInt = [ [int(round(fea[0])), int(round(fea[1]))] for fea in features ]
        
        # read 3D positions from buffer and accumulate
        for idx in range(3):
            est = positionBuffer[featuresInt[idx][1], featuresInt[idx][0]]
            if est[3] != 0:
                featureEstimates[idx] += est
        
    # average positions to obtain final estimate
    leftEye3D, rightEye3D, mouth3D = featureEstimates[:, 0:3] / featureEstimates[:, 3]

    #-------------------------------------
    print("Aligning mesh...")

    # obtain, save, and apply model matrix    
    modelMatrix = findMeshAlignment([leftEye3D, rightEye3D, mouth3D])
    np.savetxt(outDir + "modelMatrix.txt", modelMatrix)
    transformMesh(mesh, modelMatrix)

    #-------------------------------------
    # Save aligned mesh.
    # We use the pickle format, but we also save an OBJ for debugging purposes.
    print("Saving mesh...")
    meshOutputPath = os.path.join(outDir, "mesh")
    multiview.mvs_io.exportMeshAsPickle(mesh, meshOutputPath + ".pickle")
    multiview.mvs_io.exportMeshAsObj(mesh, meshOutputPath + ".obj")

    # compute and save mouth position in 3D canonical space
    newMouth3D = homogeneousTransform(modelMatrix, mouth3D)
    np.savetxt(outDir + "mouthPosition.txt", newMouth3D)

    #-------------------------------------
    # prepare VGG-face network
    print("Creating VGG-face network...")
    inputNode = tf.compat.v1.placeholder(tf.float32, [None, outres, outres, 3])
    features = vgg_face.vgg_face_features(
        dataDir + '/networks/vgg-face.mat',
        inputNode)
    vggFeats = []
    sess = tf.compat.v1.Session()

    #-------------------------------------
    # align images and cameras, and compute VGG-face features
    print("Performing image and camera alignment, and computing VGG-face features...")

    alignedCams = []
    counter = 0

    manifoldClampData = np.loadtxt(dataDir + "manifoldClamp.txt")
    validManifoldImgs = []

    for feats2D, image, camera in zip(featureCoords, images, cameras):

        if feats2D:
        
            print("Processing image %i." % counter)
            
            imgName = "img_" + str(counter).zfill(3)
            saveImage(image, outDir + imgName + "_orig.png")

            # 2D-align image
            alignmentFeats = manifold.getAlignmentFeatures(*feats2D, flipH=False)
            m = findImageAlignment(alignmentFeats)    
            warped = scipy.ndimage.affine_transform(image, m, output_shape=(outres, outres, 3), mode='nearest')
            saveImage(warped, outDir + imgName + "_aligned.png")

            # compute VGG-face features
            f = sess.run(features, feed_dict={inputNode: [warped * 255]})
            vggFeats.append(f[0])

            # export aligned cameras
            alignedCam = transformCamera(camera, modelMatrix)
            alignedCams.append(alignedCam)

            # mark images in the valid manifold range
            manifoldCoords, _ = manifold.projectToManifoldCam(alignedCam)
            clamped = manifold.clampCoordinateToManifold(manifoldCoords, manifoldClampData)    
            if np.array_equal(manifoldCoords, clamped):
                validManifoldImgs.append(counter)
            
            counter += 1

    sess.close()

    #-------------------------------------
    # save VGG-face features, cameras, and valid image indices

    print("Saving VGG-face features...")
    np.savetxt(outDir + "vggFaceFeats.txt", np.array(vggFeats))

    print("Saving cameras...")
    pickle.dump(cameras, open(outDir + "cams_orig.pickle", "wb") )
    pickle.dump(alignedCams, open(outDir + "cams_aligned.pickle", "wb") )

    print("Saving valid image indices...")
    validManifoldImgs = np.array(validManifoldImgs)
    np.savetxt(outDir + "valid.txt", validManifoldImgs)

#=======================================================

if __name__ == "__main__":
    main(sys.argv)
    print("=== TERMINATED ===")