import argparse
from fileinput import filename

import glfw
import numpy as np
import os, sys, warnings
import pickle, copy
from matplotlib import interactive
warnings.filterwarnings('ignore')

from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf

def getViewerDir():
    return os.path.dirname(os.path.realpath(__file__))

def upDir(x):
    return os.path.dirname(x)

sys.path.append(getViewerDir())
sys.path.append(upDir(getViewerDir()))

from dnnlib.util import EasyDict
import dnnlib.tflib as tflib

from graphics_utils.ogl_ops import *
from graphics_utils.render_utils import *
from graphics_utils.plot_utils import *
from graphics_utils.image_io import *
from graphics_utils.render_video import *
from cameras.camera import *

import cameras.camera_manifold as manifold
from multiview.multiview_prep import *
import multiview.mvs_io
import pipeline
from keybindings import *

# global dictionary holding all data used for rendering
renderData = EasyDict()
renderData.project_dir = os.path.join(getViewerDir())


def nowString():
    return datetime.utcnow().strftime("%Y%m%d%H%M%S")


# initialization of sliders for PCA-based semantic editing control
def initPCAEdit(pcaFile, interactive):
    renderData.pca = EasyDict()
    renderData.pca.components = np.loadtxt(pcaFile, delimiter=',')
    renderData.pca.componentCount = renderData.pca.components.shape[0]
    renderData.pca.params = np.zeros((renderData.pca.componentCount, 18), dtype=np.float32)
    renderData.pca.paramsRange = 20
    if not interactive:
        return

    def onFrameConfigure(canvas):
        canvas.configure(scrollregion=canvas.bbox("all"))

    renderData.pca.window = tk.Tk()
    renderData.pca.window.title("PCA Edit Window")
    renderData.pca.window.geometry("400x1024")
    canvas = tk.Canvas(renderData.pca.window)
    frame = tk.Frame(canvas)
    vsb = tk.Scrollbar(renderData.pca.window, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)

    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((4, 4), window=frame, anchor="nw")

    frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))

    def getValues(val):
        renderData.pca.params.fill(0)
        minStyle = rangeSliderMin.get()
        maxStyle = rangeSliderMax.get()
        if maxStyle < minStyle:
            rangeSliderMax.set(minStyle)

        for layer in range(minStyle, maxStyle + 1):
            for i, slider in enumerate(paramsSliders):
                renderData.pca.params[i, layer] = slider.get()

    def resetValues():
        for slider in paramsSliders:
            slider.set(0)

    def slider(from_, to, length, resolution=1):
        return tk.Scale(
            frame,
            from_=from_, to=to,
            orient=tk.HORIZONTAL, length=length, resolution=resolution,
            command=getValues)

    button = tk.Button(frame, text='Reset', command=resetValues)
    button.grid(row=0, column=1)

    label = tk.Label(frame, text="Min")
    label.grid(row=1, column=0)
    label = tk.Label(frame, text="Max")
    label.grid(row=2, column=0)
    rangeSliderMin = slider(0, 17, 200)
    rangeSliderMax = slider(0, 17, 200)
    rangeSliderMax.set(17)
    rangeSliderMin.grid(row=1, column=1)
    rangeSliderMax.grid(row=2, column=1)

    paramsSliders = []

    for i in range(renderData.pca.componentCount):
        label = tk.Label(frame, text=str(i))
        label.grid(row=i + 3, column=0)
        paramsSliders.append(slider(-renderData.pca.paramsRange, renderData.pca.paramsRange, 350, 0.01))
        paramsSliders[-1].grid(row=i + 3, column=1)


# initialize OpenGL shaders
def initShaders():
    superRes = tuple([renderData.multisamplingFactor * x for x in renderData.res])

    renderData.screenQuadOp = TexturedScreenQuadOP(renderData.fbo, renderData.res)
    renderData.gBufferOp = GBufferOP(renderData.fbo, renderData.res)
    renderData.texturedMeshOp = TexturedMeshOP(renderData.fbo, renderData.res)
    renderData.ulrOp = ULROP(renderData.fbo, renderData.res)
    renderData.morphologyOp = MorphologyOP(renderData.fbo, renderData.res)
    renderData.anaglyphOp = AnaglyphOP(renderData.fbo, renderData.res)
    renderData.flowOp = FlowOP(renderData.fbo, superRes, 2)
    renderData.warpingOp = WarpingOP(renderData.fbo, superRes)
    renderData.supersamplingOp = SupersamplingOP(renderData.fbo, renderData.res, 2)


# initialize render environment
def initRenderEnvironment(res):
    renderData.cam = manifold.getNeutralCam(aspect=res[0] / res[1])

    renderData.fbo = createFramebuffer()

    initShaders()

    renderData.showBlendingWeights = False
    renderData.varianceWeightToRGB = False
    renderData.lensDistortion = (0, 0)
    renderData.stereo = 0
    renderData.interocularDistance = 2
    renderData.screenDepth = 0.04
    renderData.rotationCenter = np.array((0, 0, -2), dtype=np.float32)
    renderData.changeRotationCenter = False

    renderData.outputView = None

    renderData.ganImg = np.ones((renderData.res[0], renderData.res[1], 4))
    renderData.ganTex = createRenderTargets(renderData.res, False).color

    renderData.manifoldImg = np.ones((renderData.res[0], renderData.res[0], 4))
    renderData.manifoldTex = createRenderTargets(renderData.res, False).color

    renderData.leftMouseDown = False
    renderData.rightMouseDown = False
    renderData.centerMouseDown = False
    renderData.lastClickPosition = np.zeros(2, dtype=np.float32)

    renderData.renderPath = EasyDict()
    renderData.renderPath.recording = False
    renderData.renderPath.export = False
    renderData.renderPath.step = 0
    renderData.renderPath.dir = ""
    renderData.renderPath.path = []

    renderData.inputID = 0
    renderData.manifoldViz = "Parameters"


# save current view to disk
def saveCurrentView(filename, verbose=False, resize=None):
    img = downloadImage(renderData.outputView)
    if resize is not None and img.shape[0] != resize:
        img = cv2.resize(img, dsize=(resize, resize), interpolation=cv2.INTER_AREA)
    filename = os.path.join(renderData.renderPath.dir, filename) + ".png"
    saveImage(img, filename, channels=4)
    if verbose:
        print("Saved image", filename)


# render manifold visualization figures (Parameters or top view)
def renderManifoldVisualization(manifoldCam, manifoldCoords):
    # Parameters view
    if renderData.manifoldViz == "Parameters":
        manifoldCoordsNoClamp, _ = manifold.projectToManifoldCam(renderData.cam, renderData.mouthPos)
        thetaPhiInputs = renderData.origCams.manifoldCoords[:, 0:2]
        dInputs = renderData.origCams.manifoldCoords[:, 2]
        dInputs = np.stack([dInputs, np.ones_like(dInputs)], axis=1)
        manifoldFig = scatterPlot(
            [thetaPhiInputs, dInputs],
            ranges=[renderData.manifoldVizRanges[0:2], [renderData.manifoldVizRanges[2], [0, 2]]],
            captions=["theta-phi", "d"],
            showAxes=[(True, True), (True, False)],
            flipV=(True, False),
            size=(7, 7), show=False, close=False)
        thetaPhiAxis, dAxis = manifoldFig.get_axes()
        linePlotOverlay(
            renderData.parabolas,
            renderData.manifoldVizRanges[0],
            axis=thetaPhiAxis)
        thetaPhiAxis.scatter(manifoldCoordsNoClamp[0], manifoldCoordsNoClamp[1], c='green', s=300)
        thetaPhiAxis.scatter(manifoldCoords[0], manifoldCoords[1], c='red', s=300)
        dAxis.scatter(manifoldCoords[2], 1, c='red', s=300)

    # top view
    else:
        def polar2cartProj(theta, phi, d):
            rTheta = math.radians(theta)
            rPhi = math.radians(phi)
            return d * math.cos(rPhi) * np.array([math.sin(rTheta), math.cos(rTheta)])

        def plotManifoldBoundaries(phi):
            openingAngle = min(renderData.parabolasInv[0](phi), renderData.parabolasInv[1](phi))
            endPoint = polar2cartProj(openingAngle, phi, 100)
            plt.plot([0, -endPoint[0]], [0, endPoint[1]], color="black")
            plt.plot([0, endPoint[0]], [0, endPoint[1]], color="black")

        def plotCam(cam, color, size=15):
            pos = cam.position[[0, 2]]
            viewDir = viewDirectionFromViewMatrix(cam.viewMatrix)[[0, 2]]
            viewDirOrth = np.array([viewDir[1], -viewDir[0]])
            plt.scatter(pos[0], pos[1], c=color, s=50)
            endPoint1 = pos + size * (viewDir + viewDirOrth * cam.fov)
            endPoint2 = pos + size * (viewDir - viewDirOrth * cam.fov)
            plt.plot(
                [pos[0], endPoint1[0], endPoint2[0], pos[0]],
                [pos[1], endPoint1[1], endPoint2[1], pos[1]],
                color=color)

        manifoldFig = scatterPlot(
            np.array([[0, 0]]),
            ranges=np.array([[-40, 40], [-5, 70]]),
            pointSize=400,
            captions=["Top view"],
            flipH=(True,),
            size=(7, 7), show=False, close=False)

        plotManifoldBoundaries(manifoldCoords[1])
        plotCam(renderData.cam, "green")
        plotCam(manifoldCam, "red")

    renderData.manifoldImg[...] = rasterizeFigure(manifoldFig, res=renderData.res[0])


# apply a camera transformation to the render camera
def applyCameraTransformations(t=np.eye(4), r=np.eye(4), fovChange=0):
    if not (np.array_equal(t, np.eye(4)) and np.array_equal(r, np.eye(4))):
        renderData.cam.setViewMatrix(t.dot(renderData.cam.viewMatrix.dot(r)))
    if fovChange != 0:
        newFov = renderData.cam.fov + fovChange
        renderData.cam.setProjectionParams(fov=newFov, aspect=renderData.cam.aspect)


# keyboard press or release action
def keyCallback(window, key, scancode, action, mods):
    translSpeed = 0.3
    rotSpeed = (3, 1.5, 3)
    fovSpeed = 0.01
    rotCenterSpeed = (1, 1, 1)
    interocDistSpeed = 0.1
    screenDepthSpeed = 0.0025
    t = np.eye(4)
    r = np.eye(4)
    fovChange = 0

    modeChange = False

    if action == glfw.PRESS:  # if a key is pressed

        if key == glfw.KEY_LEFT_SHIFT or key == glfw.KEY_RIGHT_SHIFT:
            renderData.HoldKeyShift = True

        if key == glfw.KEY_ESCAPE:
            print("Exiting...")
            sys.exit()

        # render mode
        if key == glfw.KEY_1:
            if not modeIs("InputView"):
                renderData.mode = "InputView"
            else:
                renderData.inputID = (renderData.inputID + 1) % renderData.origImgs.imgCount
            print("Mode:", renderData.mode, "- View:", renderData.inputID)
        elif key == glfw.KEY_2 and not modeIs("MeshFree"):
            renderData.mode = "MeshFree"
            modeChange = True
        elif key == glfw.KEY_3 and not modeIs("MeshManifold"):
            renderData.mode = "MeshManifold"
            modeChange = True
        elif key == glfw.KEY_4 and not modeIs("ULRFree"):
            renderData.mode = "ULRFree"
            modeChange = True
        elif key == glfw.KEY_5 and not modeIs("ULRManifold"):
            renderData.mode = "ULRManifold"
            modeChange = True
        elif key == glfw.KEY_6 and not modeIs("Flow"):
            renderData.mode = "Flow"
            modeChange = True
        elif key == glfw.KEY_7 and not modeIs("GANManifold") and renderData.useTF:
            renderData.mode = "GANManifold"
            modeChange = True
        elif key == glfw.KEY_8 and not modeIs("FreeStyleGAN") and renderData.useTF:
            renderData.mode = "FreeStyleGAN"
            modeChange = True
        elif (not renderData.HoldKeyShift) and key == glfw.KEY_0: # not to hold shift
            if not modeIs("ManifoldVisualization"):
                renderData.mode = "ManifoldVisualization"
            else:
                if renderData.manifoldViz == "Parameters":
                    renderData.manifoldViz = "Top"
                else:
                    renderData.manifoldViz = "Parameters"
            print("Manifold " + renderData.manifoldViz)

        if modeChange:
            print("Mode:", renderData.mode)

        if renderData.changeRotationCenter:
            # change center of rotation
            prevRotCenter = np.copy(renderData.rotationCenter)
            if key == glfw.KEY_A:
                renderData.rotationCenter[0] += rotCenterSpeed[0]
            if key == glfw.KEY_D:
                renderData.rotationCenter[0] -= rotCenterSpeed[0]
            if key == glfw.KEY_Q:
                renderData.rotationCenter[1] -= rotCenterSpeed[1]
            if key == glfw.KEY_E:
                renderData.rotationCenter[1] += rotCenterSpeed[1]
            if key == glfw.KEY_S:
                renderData.rotationCenter[2] -= rotCenterSpeed[2]
            if key == glfw.KEY_W:
                renderData.rotationCenter[2] += rotCenterSpeed[2]
            if not (renderData.rotationCenter == prevRotCenter).all():
                print("Center of rotation:", renderData.rotationCenter)
        else:
            # camera translation
            if key == glfw.KEY_A:
                t = getTranslationMatrix([translSpeed, 0, 0])
            if key == glfw.KEY_D:
                t = getTranslationMatrix([-translSpeed, 0, 0])
            if key == glfw.KEY_Q:
                t = getTranslationMatrix([0, -translSpeed, 0])
            if key == glfw.KEY_E:
                t = getTranslationMatrix([0, translSpeed, 0])
            if key == glfw.KEY_S:
                t = getTranslationMatrix([0, 0, -10 * translSpeed])
            if key == glfw.KEY_W:
                t = getTranslationMatrix([0, 0, 10 * translSpeed])

        # camera rotation
        if key == glfw.KEY_J:
            r = getRotationMatrix([rotSpeed[0], 0, 0], renderData.rotationCenter)
        if key == glfw.KEY_L:
            r = getRotationMatrix([-rotSpeed[0], 0, 0], renderData.rotationCenter)
        if key == glfw.KEY_I:
            r = getRotationMatrix([0, rotSpeed[1], 0], renderData.rotationCenter)
        if key == glfw.KEY_K:
            r = getRotationMatrix([0, -rotSpeed[1], 0], renderData.rotationCenter)
        if key == glfw.KEY_U:
            r = getRotationMatrix([0, 0, -rotSpeed[2]], renderData.rotationCenter)
        if key == glfw.KEY_O:
            r = getRotationMatrix([0, 0, rotSpeed[2]], renderData.rotationCenter)

        # zooming
        if key == glfw.KEY_M:
            fovChange = -fovSpeed
        if key == glfw.KEY_N:
            fovChange = fovSpeed

        # toggle change of camera position vs. rotation center
        if key == glfw.KEY_F:
            renderData.changeRotationCenter = not renderData.changeRotationCenter
            if renderData.changeRotationCenter:
                print("Move center of rotation")
            else:
                print("Move Camera")

        # apply camera transformations
        applyCameraTransformations(t, r, fovChange)

        # toggle lens distortion
        if key == glfw.KEY_PERIOD:
            if renderData.lensDistortion == (0, 0):
                renderData.lensDistortion = (0.25, 0)
            else:
                renderData.lensDistortion = (0, 0)

        # stereo
        if key == glfw.KEY_COMMA and modeIs("FreeStyleGAN"):
            renderData.stereo = (renderData.stereo + 1) % 4
            stereoModes = ["off", "anaglyph", "left eye", "right eye"]
            print("Stereo -", stereoModes[renderData.stereo])

        if key == glfw.KEY_LEFT_BRACKET:
            renderData.interocularDistance -= interocDistSpeed

        if key == glfw.KEY_RIGHT_BRACKET:
            renderData.interocularDistance += interocDistSpeed

        if renderData.HoldKeyShift and key == glfw.KEY_9:
            renderData.screenDepth += screenDepthSpeed

        if renderData.HoldKeyShift and key == glfw.KEY_0:
            renderData.screenDepth -= screenDepthSpeed

        # reset camera
        if key == glfw.KEY_B:
            renderData.cam = manifold.getNeutralCam()

        # snap camera to current input view
        if key == glfw.KEY_X:
            renderData.cam = copy.deepcopy(renderData.origCams.cams[renderData.inputID])

            # handle mismatching aspect ratios
            inputRes = renderData.origImgs.res[renderData.inputID]
            aspect = inputRes[1] / inputRes[0]
            s = [aspect, 1, 1] if aspect < 1 else [1, 1 / aspect, 1]
            t = np.array([1, 1, 0])
            transform = getTranslationMatrix(-t).dot(getAnisotropicScalingMatrix(s)).dot(getTranslationMatrix(t))
            renderData.cam.setProjectionMatrix(transform.dot(renderData.cam.projectionMatrix))

        # switch ULR blending weight visualization
        if key == glfw.KEY_G:
            renderData.showBlendingWeights = not renderData.showBlendingWeights

        # switch ULR variance weight visualization
        if key == glfw.KEY_V:
            renderData.varianceWeightToRGB = not renderData.varianceWeightToRGB

        # save current image
        if key == glfw.KEY_Z:
            filename = "screenshot_" + nowString()
            saveCurrentView(filename, verbose=True)

        # reload shaders
        if key == glfw.KEY_Y:
            print("Reloading shaders...")
            initShaders()
            print("Done.")

        # print key bindings
        if key == glfw.KEY_H:
            print(keyBindings)

        # -----CAMERA PATHS-------------------------------

        # start/stop recording of a camera path
        if key == glfw.KEY_R:
            if not renderData.renderPath.recording:
                renderData.renderPath.recording = True
                print("Recording camera path...")
            else:
                renderData.renderPath.recording = False
                if renderData.renderPath.path:
                    print("\nRecording finished. Saving...")
                    filetype = [('Bundler file', '*.out')]
                    dialogAnswer = filedialog.asksaveasfile(filetypes=filetype, defaultextension=filetype)
                    if dialogAnswer is not None:
                        print(
                            "Writing path (length ",
                            len(renderData.renderPath.path),
                            ") to", dialogAnswer.name, "...")
                        multiview.mvs_io.camerasToBundler(
                            dialogAnswer.name,
                            renderData.renderPath.path,
                            (renderData.res[0], renderData.res[1]))
                        print("Done")
                renderData.renderPath.path = []

        # select & play path
        if key == glfw.KEY_P:
            root = tk.Tk()
            root.withdraw()
            filetypes = [('Bundler file', '*.out'), ("PNG", "*.png")]
            pathFile = filedialog.askopenfilename(filetypes=filetypes)
            if pathFile.endswith("out"):
                renderData.renderPath.path = multiview.mvs_io.camerasFromBundler(
                    pathFile,
                    (renderData.res[0], renderData.res[1]))
                name = os.path.splitext(os.path.basename(pathFile))[0]
                renderData.renderPath.export = True
                renderData.renderPath.step = 0
                renderData.renderPath.dir = nowString() + "_" + renderData.mode + "_" + name
                renderData.renderPath.dir = os.path.join(renderData.outputDir, renderData.renderPath.dir)
                os.mkdir(renderData.renderPath.dir)
                print("Exporting path...")

    if action == glfw.RELEASE:
        if key == glfw.KEY_LEFT_SHIFT or key == glfw.KEY_RIGHT_SHIFT:
            renderData.HoldKeyShift = False


# mouse click action
def mouseClick(window, button, action, mods):
    currPos = glfw.get_cursor_pos(window)
    renderData.leftMouseDown = (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS)
    renderData.rightMouseDown = (button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS)
    renderData.centerMouseDown = (button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.PRESS)

    if renderData.leftMouseDown or renderData.rightMouseDown or renderData.centerMouseDown:
        renderData.lastClickPosition = np.array([currPos[0], currPos[1]], dtype=np.float32)


# mouse move action
def mouseMove(window, xpos, ypos):
    currPos = glfw.get_cursor_pos(window)
    translSpeed = (0.15, 0.15, 1)
    rotSpeed = (1.5, 0.75, 1.5)

    offset = (currPos - renderData.lastClickPosition) / (0.5 * renderData.res[0])
    t = np.eye(4)
    r = np.eye(4)

    if renderData.leftMouseDown:
        r = getRotationMatrix(rotSpeed * np.array([offset[0], offset[1], 0]), renderData.rotationCenter)
    if renderData.rightMouseDown:
        t = getTranslationMatrix(translSpeed * np.array([offset[0], -offset[1], 0]))
    if renderData.centerMouseDown:
        t = getTranslationMatrix(translSpeed * np.array([0, 0, -offset[1]]))

    applyCameraTransformations(t=t, r=r)


# convenience function to check the current render mode
def modeIs(modeList):
    if not isinstance(modeList, list):
        modeList = [modeList]
    return any(m in renderData.mode for m in modeList)


# the main render function
def showScreen():
    # update PCA sliders
    if renderData.useTF and renderData.showLiveScreen:
        renderData.pca.window.update()

    # path exporting
    if renderData.renderPath.export:
        renderData.cam = renderData.renderPath.path[renderData.renderPath.step]

    # cameras
    if modeIs(["MeshManifold", "ULRManifold", "Flow", "GANManifold", "FreeStyleGAN", "ManifoldVisualization"]):
        manifoldCoords, manifoldCam = manifold.projectToManifoldCam(
            renderData.cam,
            renderData.mouthPos,
            renderData.clampData if renderData.performClamping else None)
    if renderData.stereo != 0:
        leftCam, rightCam = stereoFromMonoCam(renderData.cam, renderData.interocularDistance, renderData.screenDepth)

    # path recording
    if renderData.renderPath.recording:
        recCam = manifoldCam if modeIs(["MeshManifold", "ULRManifold", "GANManifold"]) else renderData.cam
        renderData.renderPath.path.append(copy.deepcopy(recCam))
        print("\rRecording camera path: Frame", len(renderData.renderPath.path), end="")

    # render flow field(s)
    if modeIs(["Flow", "FreeStyleGAN"]):
        renderData.flow = renderData.flowOp.render(
            renderData.mesh,
            manifoldCam, renderData.cam,
            lensDistortion=renderData.lensDistortion,
            zClamp=renderData.backgroundPlaneDepth)
        if renderData.stereo:
            renderData.flowLeft = renderData.flowOp.render(
                renderData.mesh,
                manifoldCam, leftCam,
                rtID=0,
                lensDistortion=renderData.lensDistortion,
                zClamp=renderData.backgroundPlaneDepth)
            renderData.flowRight = renderData.flowOp.render(
                renderData.mesh,
                manifoldCam, rightCam,
                rtID=1,
                lensDistortion=renderData.lensDistortion,
                zClamp=renderData.backgroundPlaneDepth)

    # display input views
    if modeIs("InputView"):
        renderData.outputView = renderData.screenQuadOp.render(
            tex=renderData.origImgs.tex,
            toScreen=False,
            layer=renderData.inputID,
            transpose=renderData.origImgs.transposed[renderData.inputID])

    # render & display textured mesh
    elif modeIs(["MeshFree", "MeshManifold"]):
        meshCam = renderData.cam if modeIs("MeshFree") else manifoldCam
        lens = renderData.lensDistortion if modeIs("MeshFree") else (0, 0)
        renderData.outputView = renderData.texturedMeshOp.render(renderData.mesh, meshCam, lens)

    # render & display ULR
    elif modeIs(["ULRFree", "ULRManifold"]):
        ulrCam = renderData.cam if modeIs("ULRFree") else manifoldCam
        lens = renderData.lensDistortion if modeIs("ULRFree") else (0, 0)
        gBuffer = renderData.gBufferOp.render(renderData.mesh, ulrCam, renderData.res, lensDistortion=lens)
        ulrBuffer = renderData.ulrOp.render(
            gBuffer, ulrCam.position,
            renderData.origImgs.tex, renderData.origImgs.gBuffers, renderData.origCams.camBuffer,
            showBlendingWeights=renderData.showBlendingWeights,
            varianceWeightToRGB=renderData.varianceWeightToRGB)
        if renderData.varianceWeightToRGB:
            ulrBuffer = renderData.morphologyOp.render(ulrBuffer, kernelSize=2, erode=True, alphaOnly=False)
        renderData.outputView = ulrBuffer

    # display flow field
    elif modeIs("Flow"):
        renderData.outputView = renderData.flow

    # render & display GAN results
    elif modeIs(["GANManifold", "FreeStyleGAN"]):

        # run GAN
        renderData.feedDict[renderData.nodes.maniCoord] = manifoldCoords[None, ...]
        renderData.ganImg[..., 0:3] = renderData.sess.run(renderData.nodes.oglOutput, renderData.feedDict)[0]
        uploadImage(renderData.ganImg, renderData.ganTex)

        if modeIs("GANManifold"):
            # take GAN output as it is
            renderData.outputView = renderData.ganTex
        else:
            # warp image and perform supersampling
            def warpSuper(flow, rtIDSuper):
                warped = renderData.warpingOp.render(renderData.ganTex, flow)
                return renderData.supersamplingOp.render(warped, renderData.multisamplingFactor, rtIDSuper)

            if not renderData.stereo:
                renderData.outputView = warpSuper(renderData.flow, 0)
            else:
                if renderData.stereo == 1 or renderData.stereo == 2:
                    warpedLeftSuper = warpSuper(renderData.flowLeft, 0)
                if renderData.stereo == 1 or renderData.stereo == 3:
                    warpedRightSuper = warpSuper(renderData.flowRight, 1)
                if renderData.stereo == 1:
                    renderData.outputView = renderData.anaglyphOp.render(warpedLeftSuper, warpedRightSuper)
                elif renderData.stereo == 2:
                    renderData.outputView = warpedLeftSuper
                elif renderData.stereo == 3:
                    renderData.outputView = warpedRightSuper

    # render & display manifold visualization
    elif modeIs("ManifoldVisualization"):
        renderManifoldVisualization(manifoldCam, manifoldCoords)
        renderData.outputView = uploadImage(renderData.manifoldImg, renderData.manifoldTex)

    # put result on the screen
    if renderData.outputView is not None and renderData.showLiveScreen:
        renderData.screenQuadOp.render(tex=renderData.outputView, toScreen=True)

    # camera path export
    if renderData.renderPath.export:
        pathLength = len(renderData.renderPath.path)
        if renderData.renderPath.step < pathLength:
            filename = "path_" + str(renderData.renderPath.step).zfill(6)
            saveCurrentView(filename, resize=renderData.outputRes)
            print("\rExporting path: Frame %i/%i" % (renderData.renderPath.step + 1, pathLength), end="")
            renderData.renderPath.step += renderData.pathSubsampling
        if renderData.renderPath.step >= pathLength:
            renderData.renderPath.export = False
            renderData.cam = manifold.getNeutralCam(aspect=renderData.res[0] / renderData.res[1])
            renderData.renderPath.path = []
            if renderData.createPathVideo:
                renderVideo(renderData.outputDir + renderData.renderPath.dir, False)
            print("\nPath exported.")




# load geometry
def loadMesh(data_dir):
    renderData.mesh = pickle.load(open(data_dir + 'mesh.pickle', "rb"))
    if renderData.addPlane:
        # add a plane in the background to obtain flow values for the entire image
        addBackgroundPlane(renderData.mesh, depth=renderData.backgroundPlaneDepth)
    uploadMesh(renderData.mesh)
    renderData.mouthPos = np.loadtxt(data_dir + 'mouthPosition.txt')


# initialization of data structures for modeling the manifold boundaries
def initManifoldClampData(clampFile):
    renderData.clampData = np.loadtxt(clampFile)

    def parabola(coeffs): return lambda x: coeffs[0] * x ** 2 + coeffs[1]

    def parabolaInv(coeffs): return lambda y: math.sqrt((y - coeffs[1]) / coeffs[0])

    renderData.parabolas = []
    renderData.parabolasInv = []
    for idx in range(2):
        renderData.parabolas.append(parabola(renderData.clampData[idx]))
        renderData.parabolasInv.append(parabolaInv(renderData.clampData[idx]))


def main(argv, interactive):

    # paths
    data_dir = os.path.join(argv.model, 'freestylegan/')  # model data
    mlpFile = os.path.join(data_dir, 'model_stage2_0.pickle')
    ganFile = os.path.join(renderData.project_dir, "data/networks/stylegan2_generator.pkl")
    pcaFile = os.path.join(renderData.project_dir, "data/pca/pca_50.csv")
    clampFile = os.path.join(renderData.project_dir, "data/manifoldClamp.txt")
    renderData.outputDir = argv.output

    # render resolution
    renderData.res = (1024, 1024)

    # range of manifold visualization (theta, phi, d)
    renderData.manifoldVizRanges = np.array([[-50, 50], [-30, 30], [10, 100]], dtype=np.float32)

    # use parabolas to clamp angular manifold coordinates
    renderData.performClamping = True

    # resize input views to save some memory
    inputViewResizeFactor = 1

    # flow multisampling for anti-aliasing
    renderData.multisamplingFactor = 4

    # background plane for dense flow field
    renderData.addPlane = True
    renderData.backgroundPlaneDepth = 3.5

    # camera path export settings
    renderData.pathSubsampling = 1
    renderData.createPathVideo = False  # requires ffmpeg
    renderData.outputRes = 1024

    # enable all neural stuff
    renderData.useTF = True

    # --------------------------------------

    if renderData.res != (1024, 1024) and renderData.useTF and interactive:
        renderData.useTF = False
        print("Disabling TF because resolution is not (1024, 1024)")

    if not os.path.exists(renderData.outputDir):
        os.makedirs(renderData.outputDir)

    initManifoldClampData(clampFile)

    # Initialize the library
    print("Setting up OpenGL...")
    if interactive:
        window = oglInit(renderData.res, "FreeStyleGAN Viewer")
    else:  # make the show screen a small window in rendering mode
        window = oglInit((1, 1), "FreeStyleGAN Renderer")

    print("Loading mesh...")
    loadMesh(data_dir)

    print("Setting up render environment...")
    initRenderEnvironment(renderData.res)

    print("Preparing input views...")
    renderData.origCams, renderData.origImgs = loadViews(data_dir, resizeFactor=inputViewResizeFactor)
    renderData.origImgs.gBuffers = renderInputViewGBuffers(
        renderData.gBufferOp,
        renderData.mesh,
        renderData.origImgs,
        renderData.origCams)

    if renderData.useTF:

        print("Loading PCA controls...")
        initPCAEdit(pcaFile, interactive)

        print("Building networks...")
        tflib.init_tf()
        renderData.sess = tf.get_default_session()

        mlp = pickle.load(open(mlpFile, "rb"))
        gan = pickle.load(open(ganFile, "rb"))

        renderData.nodes = pipeline.build(gan, mlp, mlp["staticLatents"], pca=renderData.pca)

        tflib.init_uninitialized_vars()

        renderData.feedDict = {}
        renderData.feedDict[renderData.nodes.pcaComps] = renderData.pca.components[None, ...]
        renderData.feedDict[renderData.nodes.pcaParams] = renderData.pca.params[None, ...]

    if argv.mode is None:
        renderData.mode = "FreeStyleGAN" if renderData.useTF else "MeshFree"
    elif argv.mode == "ParametersManifoldViz":
        renderData.mode = "ManifoldVisualization"
        renderData.manifoldViz = "Parameters"
    elif argv.mode == "TopManifoldViz":
        renderData.mode = "ManifoldVisualization"
        renderData.manifoldViz = "Top"
    else:
        renderData.mode = argv.mode

    print("Entering main loop")

    if interactive:
        print("Press 'h' for help")
        # set GLFW event callbacks
        renderData.HoldKeyShift = False  # boolean variable for "Shift" state for controlling stereo depth using parentheses
        glfw.set_key_callback(window, keyCallback)
        glfw.set_mouse_button_callback(window, mouseClick)
        glfw.set_cursor_pos_callback(window, mouseMove)
        glfw.poll_events()
        while not glfw.window_should_close(window):  # Loop until the user closes the window
            showScreen()
            glfw.swap_buffers(window)
            glfw.poll_events()
        glfw.terminate()
    else:
        print("Running renderer!")
        renderData.renderPath.path = multiview.mvs_io.camerasFromBundler(
            argv.camera_path,
            (renderData.res[0], renderData.res[1]))
        name = os.path.splitext(os.path.basename(argv.camera_path))[0]
        renderData.renderPath.export = True
        renderData.renderPath.step = 0
        renderData.renderPath.dir = nowString() + "_" + renderData.mode + "_" + name
        renderData.renderPath.dir = os.path.join(renderData.outputDir, renderData.renderPath.dir)
        os.mkdir(renderData.renderPath.dir)
        print("Bundle file is read. Exporting path...")

        for pathStep in range(len(renderData.renderPath.path)):
            showScreen()


if __name__ == "__main__":
    msg = "using interactive viewer and rendering viewer for FreeStyleGAN"
    parser = argparse.ArgumentParser(description=msg)

    parser.add_argument("model",  help="model path directory")
    parser.add_argument("-m", "--mode", choices=["MeshFree", "MeshManiFold", "ULRFree", "ULRManifold", "GANManifold", "FreeStyleGAN", "TopManifoldViz", "ParametersManifoldViz"], help="Mode of rendering")
    parser.add_argument("-c", "--camera-path", help="camera path file")
    parser.add_argument("-o", "--output", help="the output directory for rendered images", default=os.path.join(renderData.project_dir, "results", "screenshots"))
    args = parser.parse_args()


    if args.camera_path is None:
        print("Go to interactive viewer")
        renderData.showLiveScreen = True
        main(args, interactive=True)
    elif args.camera_path.endswith("out"):
        print("Go for rendering")
        print("Output Directory:", args.output)
        renderData.showLiveScreen = False
        main(args, interactive=False)
    else:
        assert False, "Error in input arguments! camera argument should be in .out format for rendering!"

    print("=== TERMINATED ===")