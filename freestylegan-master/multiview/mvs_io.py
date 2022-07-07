import os, sys
import pywavefront
import pickle


def upDir(x):
   return os.path.dirname(x)

def getMVSDir():
    return os.path.dirname(os.path.realpath(__file__))

sys.path.append(upDir(upDir(getMVSDir())))
from dnnlib.util import EasyDict

from cameras.camera import *
from graphics_utils.image_io import *
from graphics_utils.render_utils import transformMesh
from cameras.camera import  getAnisotropicScalingMatrix
from multiview.COLMAP.colmap_parser import parseColmapData
import glob
#=======================================================

# import cameras from bundler format
# https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html
def camerasFromBundler(bundlerFile, resolution, modelMatrix=None):
    file = open(bundlerFile, 'r')
    cameras = []
    file.readline()
    camCount = int(file.readline().split()[0])
    
    # handle potentially different resolutions for MVS input views
    if not isinstance(resolution, list):
        resolution = [resolution] * camCount
    
    for i in range(camCount):
        camera = Camera()
        focalLength = float(file.readline().split()[0])
        rot = np.eye(3)
        for j in range(3):
            rot[j] = [ float(l) for l in file.readline().split() ]
        transl =  [ float(l) for l in file.readline().split() ]
        viewMatrix = np.eye(4)
        viewMatrix[0:3, 0:3] = rot
        viewMatrix[0:3, 3] = transl
        camera.setViewMatrix(viewMatrix)
        if modelMatrix is not None:
            camera = transformCamera(camera, modelMatrix)
        res = resolution[i]
        aspect = res[0] / res[1]
        fov = focalLengthToFov(focalLength, res[1])
        camera.setProjectionParams(fov, aspect)        
        cameras.append(camera)
    file.close()
    return cameras

#=======================================================

# export cameras to bundler format
# https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html
def camerasToBundler(outputFile, cameras, resolution, modelMatrix=None):
    file = open(outputFile, "w")
    file.write("# Bundle file v0.3\n")
    file.write(str(len(cameras)) + " 0\n")
    for cam in cameras:
        focalLength = fovToFocalLength(cam.fov, resolution[1])
        txt = str(focalLength) + " 0 0\n"
        if modelMatrix is not None:
            cam = transformCamera(cam, modelMatrix)
        v = cam.viewMatrix
        for r in range(3):
            for c in range(3):
                txt += str(v[r, c]) + " "
            txt += "\n"
        for c in range(3):
            txt += str(v[c, 3]) + " "
        txt += "\n"    
        file.write(txt)
    file.close()

#=======================================================

# convert cameras to lookat format
def camerasToLookat(outputFile, cameras):
    file = open(outputFile, "w")
    for idx, cam in enumerate(cameras):
        txt = "Cam" + str(idx).zfill(3) + " -D origin="
        for p in range(3):
            txt += str(cam.position[p])
            if p < 2:
                txt += ","
        v = viewDirectionFromViewMatrix(cam.viewMatrix)
        target = cam.position + v
        txt += " -D target="
        for p in range(3):
            txt += str(target[p])
            if p < 2:
                txt += ","
        up = upVectorFromViewMatrix(cam.viewMatrix)
        txt += " -D up="
        for p in range(3):
            txt += str(up[p])
            if p < 2:
                txt += ","
        txt += " -D fovy="
        txt += str(np.degrees(cam.fov))
        txt += " -D clip=1.0,1000.0"
        txt += "\n"    
        file.write(txt)
    file.close()

#=======================================================

# functions for importing data from Colmap or RealityCapture

# import scene from RealityCapture
# https://www.capturingreality.com/
def importRealityCaptureData(dataPath, useBlurred):

    #-------------------------------------------------------------- 
    # turn wavefront obj into mesh
    def expandWavefrontGeometry(geometry):
        vertices = np.array(list(geometry.materials.items())[0][1].vertices)  #vertex_format is "T2F_V3F"
        vertices = vertices.reshape((int(vertices.shape[0] / 5), 5))
        vertices = np.concatenate((vertices[..., 2:], vertices[..., :2]), axis=1)  #desired format is XYZUV
        mesh = EasyDict()
        mesh.vertices = vertices[..., :3].flatten()
        mesh.uvs = vertices[..., 3:].flatten()
        mesh.verticesAndUVs = vertices.flatten()
        mesh.faces = np.arange(int(vertices.shape[0])).flatten()
        return mesh
    #--------------------------------------------------------------

    modelPath = os.path.join(dataPath, "realitycapture", "model", "")
    registrationPath = os.path.join(dataPath, "realitycapture", "registration", "")

    # load mesh
    geometry = loadMesh(modelPath)
    mesh = expandWavefrontGeometry(geometry)

    # load texture (set it as Red if not found any)
    mesh.texture = loadTexture(modelPath, "RealityCapture")

    # load undistorted images
    images = loadUndistortedImages(registrationPath, useBlurred)

    # load camera
    cameraSetups = loadCameraSetup(registrationPath, images)

    # ignore blacklist images
    images, cameraSetups = excludeBlacklist(registrationPath, images, cameraSetups)

    return mesh, cameraSetups, images


# import scene from Colmap
# https://demuc.de/colmap/
def importColmapData(dataPath, useBlurred):

    def expandWavefrontGeometry_noUV(geometry):
        vertices = np.array(list(geometry.materials.items())[0][1].vertices)
        vertex_format = list(geometry.materials.items())[0][1].vertex_format
        if vertex_format == "V3F":
            vertices = vertices.reshape((int(vertices.shape[0] / 3), 3))
            vertices = np.concatenate((vertices[..., 0:3], np.zeros([vertices.shape[0], 2])), axis=1)  # XYZUV, set UV as zeros
        elif vertex_format == "N3F_V3F":
            vertices = vertices.reshape((int(vertices.shape[0] / 6), 6))
            vertices = np.concatenate((vertices[..., 3:6], np.zeros([vertices.shape[0], 2])), axis=1)  # XYZUV, set UV as zeros
        else:
            assert False, "vertex format of mesh does not match. It should either contain (normal+vertices) or (vertices)."
        mesh = EasyDict()
        mesh.vertices = vertices[..., :3].flatten()
        mesh.uvs = vertices[..., 3:].flatten()
        mesh.verticesAndUVs = vertices.flatten()
        mesh.faces = np.arange(int(vertices.shape[0])).flatten()
        return mesh

    modelPath = os.path.join(dataPath, "colmap", "dense", "0", "")  # including obj file
    registrationPath = os.path.join(dataPath, "colmap", "dense", "0", "images", "")  # including bundler file and images

    # parse colmap data
    parseColmapData(modelPath, registrationPath)

    # load mesh
    geometry = loadMesh(modelPath)
    mesh = expandWavefrontGeometry_noUV(geometry)

    # load texture (set it as Red if not found any)
    mesh.texture = loadTexture(modelPath, "colmap")

    # load undistorted images
    images = loadUndistortedImages(registrationPath, useBlurred)

    # load camera
    cameraSetups = loadCameraSetup(registrationPath, images)

    #modify Y and Z axes for mesh, camera, and viewMatrix
    trans_matrix1 = getAnisotropicScalingMatrix([1, -1, -1])  # Flip Y and Z for mesh and camera
    trans_matrix2 = getAnisotropicScalingMatrix([1, -1, -1])  # Flip Y and Z for viewMatrix
    transformMesh(mesh, trans_matrix1)
    for idx in range(len(cameraSetups)):
        cameraSetups[idx] = transformCamera(cameraSetups[idx], trans_matrix1)
        cameraSetups[idx].setViewMatrix(trans_matrix2.dot(cameraSetups[idx].viewMatrix))

    # ignore blacklist images
    images, cameraSetups = excludeBlacklist(registrationPath, images, cameraSetups)

    return mesh, cameraSetups, images


def loadMesh(modelPath):
    mesh_files = glob.glob(os.path.join(modelPath, "*.obj"))
    assert not len(mesh_files) == 0, "No obj mesh file found"
    assert not len(mesh_files) >= 2, "Only one obj Mesh file should be found in this folder"
    return pywavefront.Wavefront(mesh_files[0], collect_faces=True)


def loadTexture(modelPath, mode):
    if mode == "RealityCapture":
        textureFile = None
        for file in os.listdir(modelPath):
            if file.endswith("png"):
                textureFile = file
                break
        assert textureFile is not None, "RealityCapture data needs a texture file. No png texture file found!"
        return loadImage(modelPath + textureFile)
    elif mode == "colmap":
        texture = np.zeros([100, 100, 3], dtype=np.float32)  # make an arbitrary texture
        texture[:, :, 0] = 1  # set the texture as red
        return texture
    assert False, "mode is wrong for loading the texture!"


def loadCameraSetup(registrationPath, images):
    cameraFile = None
    for file in os.listdir(registrationPath):
        if file.endswith("out"):
            cameraFile = file
            break
    assert cameraFile is not None, "No bundler camera file found"

    cameraSetups = camerasFromBundler(
        registrationPath + cameraFile,
        [(img.shape[1], img.shape[0]) for img in images])

    return cameraSetups


def excludeBlacklist(registrationPath, images, cameraSetups):
    blacklistFile = registrationPath + 'blacklist.txt'
    blacklist = []
    if os.path.isfile(blacklistFile):
        blacklist = np.loadtxt(blacklistFile, dtype=int, ndmin=1)

    if len(blacklist) != 0:
        imgsTmp = []
        camsTmp = []
        for idx, (img, cam) in enumerate(zip(images, cameraSetups)):
            if idx not in blacklist:
                imgsTmp.append(img)
                camsTmp.append(cam)
        images = imgsTmp
        cameraSetups = camsTmp
    return images, cameraSetups


def loadUndistortedImages(registrationPath, useBlurred=False):
    # find the format of images in the directory
    if useBlurred is True:  # as the output of blur_background script is .png
        formatSuffix = ".png"
    else:
        formatSuffix = detectImageFormat(registrationPath)

    # load images
    imageFiles = []
    blurSuffix = "_blur"
    matteSuffix = "_matte"

    for file in sorted(os.listdir(registrationPath)):
        if useBlurred:
            if file.endswith(blurSuffix + formatSuffix):
                imageFiles.append(file)
        else:
            if file.endswith(formatSuffix) and not file.endswith(blurSuffix + formatSuffix) and not file.endswith(
                    matteSuffix + formatSuffix):
                imageFiles.append(file)
    assert len(imageFiles) != 0, "No image file found with suffix " + formatSuffix
    images = []
    for file in imageFiles:
        imagepath = registrationPath + file
        images.append(loadImage(imagepath))

    return images

#=======================================================

# export mesh using python pickle to speed up loading
def exportMeshAsPickle(mesh, filename):
    revert = mesh.gpu
    if revert:
        vao = mesh.vao
        textureGPU = mesh.textureGPU
        mesh.vao = None
        mesh.textureGPU = None
        mesh.gpu = False
    pickle.dump(mesh, open(filename, "wb") )
    if revert:
        mesh.vao = vao
        mesh.textureGPU = textureGPU
        mesh.gpu = True

#=======================================================

# export mesh as OBJ
def exportMeshAsObj(mesh, filename):
    hasUVs = mesh.uvs is not None
    verts = np.reshape(mesh.vertices, (int(mesh.vertices.shape[0] / 3), 3))
    if hasUVs:
        uvs = np.reshape(mesh.uvs, (int(mesh.uvs.shape[0] / 2), 2))
    faces = np.reshape(mesh.faces, (int(mesh.faces.shape[0] / 3), 3)) + 1
    with open(filename, 'w') as file:
        for v in verts:
            line = "v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2])
            file.write(line + "\n")
        if hasUVs:
            for c in uvs:
                line = "vt " + str(c[0]) + " " + str(c[1])
                file.write(line + "\n")
        for f in faces:
            if hasUVs:
                a, b, c = [str(ff) for ff in f]
                line = "f " + a + "/" + a + " " +  b + "/" + b + " " + c + "/" + c
            else:
                line = "f " + str(f[0]) + " " + str(f[1]) + " " + str(f[2])
            file.write(line + "\n")
        file.close()