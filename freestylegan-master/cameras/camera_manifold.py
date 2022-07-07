import sys, os
import numpy as np
from scipy.optimize import root

def getCameraDir():
    return os.path.dirname(os.path.realpath(__file__))

def upDir(x):
   return os.path.dirname(x)

sys.path.append(getCameraDir())

from camera import *
import alignment_coords

#=======================================================

# define a "neutral" (i.e., frontal) camera for initialization
def getNeutralCam(fov=0.35, aspect=1):
    neutralCam = Camera()
    neutralDst = 25
    neutralCam.setViewMatrix(getTranslationMatrix(np.array([0, 0, -neutralDst])))
    neutralCam.setProjectionParams(fov=fov, aspect=aspect)
    return neutralCam

#=======================================================

# convert a manifold coordinate to a perspective camera
def manifoldCoordToCamera(manifoldCoords, mouth3D=None, aspect=1):
    c = Camera()
    r = getRotationMatrix([manifoldCoords[0], manifoldCoords[1], 0])
    t = getTranslationMatrix([0, 0, -manifoldCoords[2]])
    c.setViewMatrix(t.dot(r))
    if mouth3D is not None:
        _, c = optimizeManifoldCam(c, mouth3D, aspect)
    return c

#=======================================================

# obtain x_- and x_+
def getAlignmentFeatures(eyeLeft, eyeRight, mouth, flipH=True):
    eye_avg = (eyeLeft + eyeRight) * 0.5
    eye_to_eye = eyeRight - eyeLeft
    eye_to_mouth = mouth - eye_avg
    eye_to_mouth_flip = np.flipud(eye_to_mouth) * [-1, 1]
    if flipH:
        eye_to_mouth_flip *= -1
    x = eye_to_eye - eye_to_mouth_flip
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    center = eye_avg + eye_to_mouth * 0.1
    return [center - x, center + x]

#=======================================================

# project a camera to the manifold (no boundary clamping)
def optimizeManifoldCam(camera, mouth3D, aspect=1):
    
    def objective(params):
        alpha, beta, gamma, fov = params
        rotation = getRotationMatrix([alpha, beta, gamma])
        projection = getProjectionMatrix(fov)
        fullTransform = projection.dot(rotation).dot(camera.viewMatrix)
        eyeLeft = perspDiv2D(fullTransform.dot(alignment_coords.eyes3D[0]))
        eyeRight = perspDiv2D(fullTransform.dot(alignment_coords.eyes3D[1]))
        mouth = perspDiv2D(fullTransform.dot(np.array([*mouth3D, 1])))
        feats = getAlignmentFeatures(eyeLeft, eyeRight, mouth)
        diff = np.array(feats) - np.array(alignment_coords.referenceAlignmentFeats)
        return [*diff.flatten()]
     
    optRes = root(objective, [0, 0, 0, camera.fov], method="lm")
    
    if optRes.success:
        alpha, beta, gamma, fov = optRes.x
        camera.setProjectionParams(fov, aspect)
        manifoldRotation = getRotationMatrix([alpha, beta, gamma])
        camera.setViewMatrix(manifoldRotation.dot(camera.viewMatrix))

    return optRes.success, camera

#=======================================================

# obtain manifold coordinates from camera position
def manifoldCoordFromPosition(p):
    #---------------------------------------
    def getAngle(v):
        dotV = v[-1] / np.linalg.norm(v)
        angle = np.degrees(np.arccos(dotV))
        if v[0] > 0:
            angle *= -1
        return angle
    #---------------------------------------
    
    maniCoords = np.zeros(3)
    maniCoords[0] = getAngle(p[[0, 2]])
    maniCoords[1] = -getAngle(p[[1, 2]])
    maniCoords[2] = np.linalg.norm(p)
    return maniCoords

#=======================================================

# project a camera to the manifold (with boundary clamping)
def projectToManifoldCam(camera, mouth3D=None, clampData=None):
    p = camera.position
    maniCoords = manifoldCoordFromPosition(p)
    if clampData is not None:
        maniCoords = clampCoordinateToManifold(maniCoords, clampData)
    cam = None
    if mouth3D is not None:
        cam = manifoldCoordToCamera(maniCoords, mouth3D, camera.aspect)
    return maniCoords, cam

#==============================================

# map manifold coordinates to/from a range
def normalizeManifoldCoords(maniCoord, ranges, inverse=False):
    #----------------------------
    def normalize(c):
        for i in range(ranges.shape[0]):
            if inverse:
                c[i] = (ranges[i,1] - ranges[i,0]) / 2 * (c[i] + 1) + ranges[i,0]
            else:
                c[i] = 2 / (ranges[i,1] - ranges[i,0]) * (c[i] - ranges[i,0]) - 1
        return c
    #----------------------------
    mOut = np.copy(maniCoord)
    if maniCoord.ndim == 1:
        return normalize(mOut)
    else:
        for idx in range(maniCoord.shape[0]):
            mOut[idx] = normalize(mOut[idx])
    return mOut

#==============================================

# clamp manifold coordinates to valid range
def clampCoordinateToManifold(query, clampData):
    #------------------------------
    def evalCurve(x, c):
        return c[0] * x ** 2 + c[1]
    #------------------------------
    def robustCubeRoot(x):
        third = 1. / 3
        return x ** third if x >= 0 else -((-x) ** third)
    #------------------------------
    def closestPoint(query, c):
        denom = 2 * c[0] ** 2
        p = (1 + 2 * c[0] * (c[1] - query[1])) / denom / 3
        q = -query[0] / denom / 2
        D = (p ** 3 + q ** 2) ** (1. / 2)
        x = robustCubeRoot(-q + D) + robustCubeRoot(-q - D)
        return np.array([x, evalCurve(x, c)])
    #------------------------------
    
    thetaPhiCurves = clampData[0:2]
    dClamp = clampData[2]

    # clamp d
    d = np.clip(query[2], dClamp[0], dClamp[1])

    # check conditions
    uc, lc = thetaPhiCurves
    belowUc = evalCurve(query[0], uc) > query[1]
    aboveLc = evalCurve(query[0], lc) < query[1]
    
    # point lies between the curves
    if belowUc and aboveLc:
        return np.array([*query[:2], d])
    
    # point lies outside of both curves
    if not belowUc and not aboveLc:
        curveCross = ((lc[1] - uc[1]) / (uc[0] - lc[0])) ** (1. / 2)
        curveCross *= np.sign(query[0])
        return np.array([curveCross, evalCurve(curveCross, uc), d])
    
    # point lies above both curves
    if not belowUc:
        return np.array([*closestPoint(query[:2], uc), d])
    
    # point lies below both curves
    if not aboveLc:
        return np.array([*closestPoint(query[:2], lc), d])
    
#==============================================

# generate a valid manifold sample by numerically inverting the CDF of c_u - c_l
def sampleManifold(clampData, useAABB=False, scaleAABB=1):
    deltaA = clampData[0,0] - clampData[1,0]
    deltaB = clampData[0,1] - clampData[1,1]
    clampD = clampData[2]

    #------------------------------------    
    def rescale(x, low, high):
        return x * (high - low) + low
    #------------------------------------    

    randX, randY, randZ = np.random.random(3).astype(np.float32)
    
    if useAABB:
        intersect = ((clampData[1, 1] - clampData[0, 1]) / (clampData[0, 0] - clampData[1, 0])) ** (1. / 2)
        theta = scaleAABB * rescale(randX, -intersect, intersect)
        phi = scaleAABB * rescale(randY, clampData[1][1], clampData[0][1])
    else:
        #------------------------------------
        def evalCurve(x, c):
            return c[0] * x ** 2 + c[1]
        #------------------------------------
        def cdf(x):
            normalization = 2 * (deltaA / 3 * (-deltaB / deltaA) ** (3/2) + deltaB * math.sqrt(-deltaB / deltaA))
            return (deltaA / 3 * x**3 + deltaB * x) / normalization + 0.5
        #------------------------------------
        def inversionObjective(param):
            return cdf(param) - randX
        #------------------------------------    
    
        optRes = root(inversionObjective, 0)
        theta = optRes.x[0]

        cu = evalCurve(theta, clampData[0])
        cl = evalCurve(theta, clampData[1])

        phi = rescale(randY, cl, cu)
    
    d = rescale(randZ, clampD[0], clampD[1])
    return [theta, phi, d]


#==============================================
#==============================================
# Below: test and visualization code only
#==============================================
#==============================================

# visualize the clamping of manifold coordinates to the valid range
def visualizeManifoldClamping():

    sys.path.append(upDir(getCameraDir()))

    import matplotlib.pyplot as plt
    import graphics_utils.plot_utils
    
    #------------------------

    curvePath = "../data/manifoldClamp.txt"
    angleRange = np.array([[-60., 60.], [-25., 40.]])
    testPointCount = 250
    vizPath = "../data/manifoldClampViz.pdf"

    #------------------------

    clampData = np.loadtxt(curvePath)

    # create test points
    testPoints = (2 * np.random.random((testPointCount, 2)).astype(np.float32)) - 1
    testPoints = normalizeManifoldCoords(testPoints, angleRange, inverse=True)    

    # perform projection
    projections = np.copy(testPoints)
    for idx, p in enumerate(testPoints):
        p = np.array([*p, 8])
        pClamp = clampCoordinateToManifold(p, clampData)
        projections[idx] = pClamp[0:2]

    #---------PLOTS-----------------------------------

    fig = graphics_utils.plot_utils.scatterPlot(
        testPoints,
        size=(15, 12),
        pointSize=10,
        ranges=angleRange,
        show=False, close=False)

    for idx in range(testPointCount):
        fig.gca().plot(
            [testPoints[idx, 0], projections[idx, 0]],
            [testPoints[idx, 1], projections[idx, 1]],
            lw=0.5)


    def parabola(coeffs):
        return lambda x : coeffs[0] * x ** 2 + coeffs[1]

    fcts = []
    for idx in range(2):
        fcts.append(parabola(clampData[idx]))

    graphics_utils.plot_utils.linePlotOverlay(fcts, angleRange[0], linewidth=2)

    plt.savefig(vizPath)

#==============================================

# visualize manifold sampling
def testManifoldSampling():

    sys.path.append(upDir(getCameraDir()))

    import matplotlib.pyplot as plt
    import graphics_utils.plot_utils

    #--------------------------------------

    testPointCount = 1000
    angleRange = np.array([[-40., 40.], [-20., 25.]])

    curvePath = "../data/manifoldClamp.txt"
    vizPath = "../data/manifoldSamplingViz.pdf"
    
    #--------------------------------------
    
    clampData = np.loadtxt(curvePath)

    testPoints = np.zeros((testPointCount, 2), dtype=np.float32)

    for idx, p in enumerate(testPoints):
        testPoints[idx] = sampleManifold(clampData)[0:2]


    fig = graphics_utils.plot_utils.scatterPlot(
        testPoints,
        size=(4, 4),
        pointSize=0.2,
        ranges=angleRange,
        show=False, close=False)

    def parabola(coeffs):
        return lambda x : coeffs[0] * x ** 2 + coeffs[1]

    fcts = []
    for idx in range(2):
        fcts.append(parabola(clampData[idx]))

    graphics_utils.plot_utils.linePlotOverlay(fcts, angleRange[0], linewidth=1)
    
    plt.savefig(vizPath)

#==============================================

if __name__ == "__main__":    
    visualizeManifoldClamping()
    testManifoldSampling()
    print("=== TERMINATED ===")