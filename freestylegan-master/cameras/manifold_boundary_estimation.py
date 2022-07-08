import os, sys
import face_alignment  # from https://github.com/1adrianb/face-alignment
import matplotlib.pyplot as plt
import collections
from scipy.optimize import root
import cv2

def getCameraDir():
    return os.path.dirname(os.path.realpath(__file__))

def upDir(x):
   return os.path.dirname(x)

sys.path.append(getCameraDir())
sys.path.append(upDir(getCameraDir()))

from graphics_utils.image_io import *
from graphics_utils.plot_utils import *
from cameras.camera import getRotationMatrixAxisAngle
from cameras.camera_manifold import normalizeManifoldCoords

#=======================================================

# Associate facial features with their meaning and visualization color
def getFeatureSlices():
    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    slices = {  'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4)),
                'lipsCenterTop': pred_type(slice(50, 53), (1., 0, 0, 1.)),
                'lipsCenterBottom': pred_type(slice(56, 59), (1., 0, 0, 1.))
    }
    return slices

#=======================================================

# aggregate facial features into representative eye and mouth locations
def aggregateFeatures(preds, slices):
    eye1 = np.array(preds[slices['eye1'].slice])
    eye2 = np.array(preds[slices['eye2'].slice])
    lips = np.array(preds[slices['lipsCenterTop'].slice])
    eye1_mean = np.mean(eye1, axis=0)
    eye2_mean = np.mean(eye2, axis=0)
    mouth_mean = np.mean(lips, axis=0)
    return eye1_mean, eye2_mean, mouth_mean

#=======================================================

# visualize facial recognition features in 2D and 3D
def plotFeatures(img, preds, slices, show=False, filename=None):
      
    # 2D-Plot
    plot_style = dict(marker='o',
                    markersize=4,
                    linestyle='-',
                    lw=2)

    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img)

    for s in slices.values():
        ax.plot(preds[s.slice, 0],
                preds[s.slice, 1],
                color=s.color, **plot_style)

    ax.axis('off')

    # 3D-Plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.scatter(preds[:, 0], 
                    preds[:, 1],
                    preds[:, 2],
                    c='cyan',
                    alpha=1.0,
                    edgecolor='b')

    for s in slices.values():
        ax.plot3D(preds[s.slice, 0],
                preds[s.slice, 1],
                preds[s.slice, 2], color='blue')

    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])
    
    plt.tight_layout()

    if show:
        plt.show()
    if filename is not None:
        plt.savefig(filename)
        plt.close(fig)
    
#=======================================================

# plot 2D angular distribution of manifold coordinates
def plotCoordDistribution(outFile, manifoldCoords, angleRange, curves=None):
    print("Plotting distribution...")
    fig = scatterPlot(
        manifoldCoords,
        size=(15, 12),
        pointSize=5,
        ranges=angleRange,
        captions=["Manifold coordinate distribution"],
        show=False, close=False)
    ax = fig.gca()
    if curves is not None:
        curveSamplesX = np.linspace(angleRange[0,0], angleRange[0,1], 100)
        for c in curves:
            curveSamplesY = c[0] * curveSamplesX * curveSamplesX + c[1]
            ax.plot(curveSamplesX, curveSamplesY)

    plt.savefig(outFile)

#=======================================================

# convert 3D eye and mouth positions into manifold coordinates
def featuresToManifoldCoord(eye1, eye2, mouth):

    eye_mid = (eye1 + eye2) / 2
    eyeDst = np.linalg.norm(eye2 - eye1)

    # Find horizontal rotation
    eyeDstX = eye2[0] - eye1[0]
    theta = np.degrees(np.arccos(eyeDstX / eyeDst))
    if eye1[2] < eye2[2]:
        theta *= -1

    # Find vertical rotation:
    # How much do we need to rotate mouth around eye axis to achieve 0.125-times inter-ocular distance from eye plane?
    rotAxis = (eye2 - eye1) / eyeDst
    planeNormal = np.cross(rotAxis, np.array([0,1,0]))

    def objective(angle):

        # rotate mouth around eye axis
        r = getRotationMatrixAxisAngle(rotAxis, angle, eye_mid)
        m = r.dot(np.array([*mouth, 1]))
        
        # measure distance between rotated mouth and vertical plane through eyes
        d = planeNormal.dot(m[0:3] - eye_mid)

        # penalize distance different from quarter of inter-ocular distance
        return d - 0.125 * eyeDst 

    optRes = root(objective, 0)
    phi = optRes.x[0]

    return theta, phi

#=======================================================

# detect facial features and convert to manifold coordinates
def detectPoses(imgFiles, vizPath):
    # set up detector
    print("Loading detector network...")
    detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)
    slices = getFeatureSlices()

    # iterate over all images
    manifoldCoords = []
    for idx, imgFile in enumerate(imgFiles):
        print("\rProcessing image %i/%i" % (idx+1, len(imgFiles)), end="")
        img = loadImage(imgFile, normalize=False)[..., 0:3]
        feats = detector.get_landmarks_from_image(img)
        preds = feats[-1] if feats else None
        if preds is None:
            continue
        eye1, eye2, mouth = aggregateFeatures(preds, slices)
        coords = featuresToManifoldCoord(eye1, eye2, mouth) 
        manifoldCoords.append(coords)
        if vizPath is not None:
            vizFile = os.path.join(vizPath, "viz_" + str(idx).zfill(6) + ".png")
            plotFeatures(img, preds, slices, filename=vizFile, show=False)
    print("")
    manifoldCoords = np.array(manifoldCoords)
    return manifoldCoords

#=======================================================

# conversion of manifold coordinates from/to pixels
def manifoldToPixel(m, angleRange, res):
    cNorm = normalizeManifoldCoords(m, angleRange)
    cNorm[:,1] *= -1
    cNorm = 0.5 * (cNorm + 1)
    return cNorm * res

def pixelToManifold(p, angleRange, res):
    c = (2. * p / res) - 1
    c[:,1] *= -1
    return normalizeManifoldCoords(c, angleRange, inverse=True)

#=======================================================

# perform density estimation of manifold coordinates
def densityEstimation(manifoldCoords, res, sigma, angleRange):
    
    print("Density estimation...")

    densityCanvas = np.zeros(res[::-1], dtype=np.float32)
    sigmaSqr = sigma * sigma
    splatSize = 3 * sigma

    m = manifoldToPixel(manifoldCoords, angleRange, res)

    for idx, p in enumerate(m):
        if p[0] < 0 or p[0] > res[0] or p[1] < 0 or p[1] > res[1]:
            continue
        for y in range(-splatSize, splatSize):
            for x in range(-splatSize, splatSize):
                if p[1]+y < 0 or p[1]+y > res[1] or p[0]+x < 0 or p[0]+x > res[0]:
                    continue
                d = x*x + y*y
                w = np.exp(-d / (2*sigmaSqr))
                densityCanvas[int(p[1]+y), int(p[0]+x)] += w        
        if idx % 100 == 0:
            print("\r%i/%i" % (idx, manifoldCoords.shape[0]), end="")
    
    print("")
    densityCanvas /= np.max(densityCanvas)    
    return densityCanvas

#=======================================================

# extract upper and lower edges of binarized density
def findDistributionBoundaries(densityCanvas, densityThreshold, angleRange, res, path=None):
    
    def getEdgeSamples(e):
        e = np.stack(np.where(e == 1), axis=-1)[:, [1,0]]
        return pixelToManifold(e, angleRange, res)
    
    thrDensity = np.where(densityCanvas > densityThreshold, 1., 0.).astype(np.float32)
    kernel = np.array([[0, 1, -1]], dtype=np.float32)
    edgesDensityY = cv2.filter2D(thrDensity, cv2.CV_32F, kernel=np.transpose(kernel))

    upperEdge = np.clip(-edgesDensityY, 0, 1)
    lowerEdge = np.clip(edgesDensityY, 0, 1)

    if path is not None:
        edgeViz = np.stack([upperEdge, lowerEdge, np.zeros_like(upperEdge)], axis=-1)
        saveImage(edgeViz, os.path.join(path, "densityEdges.png"))

    return getEdgeSamples(upperEdge), getEdgeSamples(lowerEdge)

#=======================================================

# least-squares fit of samples to parabola: a * x^2 + b
def fitParabola(edgeSamples):
    
    # build linear system
    A = np.ones((edgeSamples.shape[0], 2), dtype=np.float32)
    A[:,0] = np.square(edgeSamples[:, 0])
    rhs = edgeSamples[:,1]
    At = np.transpose(A)

    # solve it using pseudo-inverse
    pInv = np.linalg.inv(At.dot(A)).dot(At)
    return pInv.dot(rhs)

#=======================================================

# run boundary estimation
def main(argv):

    path = argv[1]
    outputPath = os.path.join(path, "boundary_estimation/")
    
    # Number of images to process. Set to -1 to consider all images in path.
    maxImgs = -1
    
    # The manifold boundaries will be stored here.
    curvesOutputFile = os.path.join(outputPath, "manifoldClamp.txt")
    
    # Enable visualization of facial features for all images.
    outputFeatureViz = False

    # Range of theta-phi angles to consider.
    angleRange = np.array([[-40., 40.], [-30., 30.]])
    
    # Clamp the d component.
    dClamp = np.array([10., 100.], dtype=np.float32)

    # params: density estimation
    densityEstimationRes = 128
    densityEstimationSigma = 3
    densityThreshold = 0.01
    
    #-------------------------------

    imgFiles = findImages(path)
    if maxImgs != -1:
        imgFiles = imgFiles[:min(maxImgs, len(imgFiles))]

    if outputPath is not None:
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)

    # analyze head poses and save estimated manifold coords
    manifoldCoords = detectPoses(imgFiles, outputPath if outputFeatureViz else None)
    np.savetxt(os.path.join(outputPath, "manifoldCoords.txt"), manifoldCoords)
    plotCoordDistribution(os.path.join(outputPath, "scatter.png"), manifoldCoords, angleRange)

    # density estimation
    rangeAspect = (angleRange[0][1] - angleRange[0][0]) / (angleRange[1][1] - angleRange[1][0])
    dRes = (densityEstimationRes, int(densityEstimationRes / rangeAspect))
    densityCanvas = densityEstimation(manifoldCoords, dRes, densityEstimationSigma, angleRange)
    saveImage(densityCanvas, os.path.join(outputPath, "density.png"))
    
    # fit curves to distribution boundaries
    upperEdgeSamples, lowerEdgeSamples = findDistributionBoundaries(
        densityCanvas, densityThreshold, angleRange, dRes, outputPath)
    curves = [fitParabola(upperEdgeSamples), fitParabola(lowerEdgeSamples), dClamp]
    np.savetxt(curvesOutputFile, curves)
    plotCoordDistribution(os.path.join(outputPath, "scatter+fit.png"), manifoldCoords, angleRange, curves=curves)
        
#=======================================================

if __name__ == "__main__":    
    main(sys.argv)
    print("=== TERMINATED ===")




