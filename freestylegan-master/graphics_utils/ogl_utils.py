import os, sys

import glfw
glfw_ok = glfw.init()  # trying to open GLFW. If cannot, EGL variable should be set to initialize it in oglInit().
if not glfw_ok:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Must be set before importing OpenGL.

import importlib
from OpenGL.GL import *
import OpenGL
from OpenGL.GL.shaders import compileProgram, compileShader
import OpenGL.error
import numpy as np
import copy, ctypes

#=======================================================

def getUtilsDir():
    return os.path.dirname(os.path.realpath(__file__))

def upDir(x):
   return os.path.dirname(x)

sys.path.append(getUtilsDir())
sys.path.append(upDir(upDir(getUtilsDir())))

from dnnlib.util import EasyDict

#=======================================================

def init_egl():
    import OpenGL.EGL as egl
    import ctypes
    display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
    assert display != egl.EGL_NO_DISPLAY, "cannot receive Display"

    major = ctypes.c_int32()  # type of int32
    minor = ctypes.c_int32()
    ok = egl.eglInitialize(display, major, minor)  # initialize the display and return minor and major version number of EGL implementation. major = c_int(1), minor = c_int(5)
    assert ok, "EGL initialization cannot be performed!"

    # Choose config.
    config_attribs = [
        egl.EGL_RENDERABLE_TYPE, egl.EGL_OPENGL_BIT,  # creating OpenGL contexts
        egl.EGL_SURFACE_TYPE, egl.EGL_PBUFFER_BIT,  # creating pixel buffer surfaces
        egl.EGL_NONE
    ]
    configs = (ctypes.c_int32 * 1)()  # need only one configuration to be returned in eglChooseConfig
    num_configs = ctypes.c_int32()
    ok = egl.eglChooseConfig(display, config_attribs, configs, 1, num_configs)  # return a frame buffer configuration based on required attributions in config_attribs
    assert ok
    assert num_configs.value == 1
    config = configs[0]  # EGL frame buffer configuration

    # Create dummy pbuffer surface.
    surface_attribs = [
        egl.EGL_WIDTH, 1,
        egl.EGL_HEIGHT, 1,
        egl.EGL_NONE
    ]
    surface = egl.eglCreatePbufferSurface(display, config, surface_attribs)  # create an off-screen pBuffer
    assert surface != egl.EGL_NO_SURFACE

    # Setup GL context.
    ok = egl.eglBindAPI(egl.EGL_OPENGL_API)  # set the rendering API as EGL
    assert ok
    context = egl.eglCreateContext(display, config, egl.EGL_NO_CONTEXT, None)  # use the frame buffer configuration to create rendering context without sharing with other contexts (and we don't have any)
    assert context != egl.EGL_NO_CONTEXT
    ok = egl.eglMakeCurrent(display, surface, surface, context)  # connect the content to surface pBuffer
    assert ok
    return ok

#=======================================================

# initialize OpenGL context
def oglInit(res=(5,5), title="OpenGL Window"):
    if not glfw_ok:
        print("Unable to init GLFW. trying for EGL...")
        if not init_egl():
            assert False, "Unable to initialize OpenGL!!"
        print("EGL initialized successfully.")
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(res[0], res[1], title, None, None)
    assert window, "Unable to create GLFW window"
    glfw.make_context_current(window)
    print("OpenGL initialized successfully.")
    return window

#=======================================================

# create a framebuffer
def createFramebuffer(n=1):
    return glGenFramebuffers(n)

#=======================================================

# delete a framebuffer
def deleteFramebuffer(fbo):
    glDeleteFramebuffers(1, [fbo])

#=======================================================

# create a shader program, consisting of vertex and fragment shaders
def createShader(pathVertex, pathFragment):
    vertexSrc = open(pathVertex + ".vert", "r").read()
    fragmentSrc = open(pathFragment + ".frag", "r").read()
    shader = compileProgram(compileShader(vertexSrc, GL_VERTEX_SHADER), compileShader(fragmentSrc, GL_FRAGMENT_SHADER))
    return shader

#=======================================================

# create a texture
def createTexture(n=1):
    return glGenTextures(n)

#=======================================================

# delete a texture
def deleteTexture(t):
     glDeleteTextures(1, [t])

#=======================================================

# create render targets for rendering to a texture
def createRenderTargets(res, createDepthBuffer, channelCount=4):

    if isinstance(res, int):
        res = (res, res)
    
    rts = EasyDict()
    rts.color = createTexture()
    glBindTexture(GL_TEXTURE_2D, rts.color)
    glFormat = glFormatFromChannelCount(channelCount)
    glTexImage2D(GL_TEXTURE_2D, 0, glFormat[0], res[0], res[1], 0, glFormat[1], GL_FLOAT, None)
    if createDepthBuffer:
        rts.depth = createTexture()
        glBindTexture(GL_TEXTURE_2D, rts.depth)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, res[0], res[1], 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    return rts

#=======================================================

# delete render targets
def deleteRenderTargets(rts):
    if rts.color is not None:
        deleteTexture(rts.color)
    if rts.depth is not None:
        deleteTexture(rts.depth)

#=======================================================

# build a linear MIP map
def buildMIP(tex):
    glBindTexture(GL_TEXTURE_2D, tex)
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)

#=======================================================

# set texture sampling parameters
def setTexParams(target= GL_TEXTURE_2D, minFilter=GL_NEAREST, magFilter=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER):
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minFilter)
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, magFilter)
    glTexParameterfv(target, GL_TEXTURE_BORDER_COLOR, [0, 0, 0, 1])
    glTexParameteri(target, GL_TEXTURE_WRAP_S, wrap)
    glTexParameteri(target, GL_TEXTURE_WRAP_T, wrap)

#=======================================================

# get default OpenGL (internal) format from channel count
def glFormatFromChannelCount(c):
    if c == 1:
        return GL_R32F, GL_RED
    if c == 2:
        return GL_RG32F, GL_RG
    if c == 3:
        return GL_RGB32F, GL_RGB
    if c == 4:
        return GL_RGBA32F, GL_RGBA

#=======================================================

# upload an image to the GPU
def uploadImage(image, texture=None, flipH=True):
    if image.ndim == 2:
        image = image[..., None]
    w, h, c = image.shape
    if flipH:
        image = np.flip(image, 0)
    if texture is None:
        texture = createTexture()
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glFormat = glFormatFromChannelCount(c)
    glTexImage2D(GL_TEXTURE_2D, 0, glFormat[0], h, w, 0, glFormat[1], GL_FLOAT, image)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture

#=======================================================

# upload a list of images to a texture array
def uploadImageList(imgList, textureArray=None, flipH=True):
    imgCount = len(imgList)
    assert imgCount > 0, "List is empty."
    w, h, c = imgList[0].shape
    _imgList = copy.deepcopy(imgList)
    for i in range(imgCount):
        assert imgList[0].shape == imgList[i].shape, "Image dimensions don't match."
        if flipH:
            _imgList[i] = np.flip(_imgList[i], 0)
    if textureArray is None:
        textureArray = createTexture()
    glBindTexture(GL_TEXTURE_2D_ARRAY, textureArray)
    glinternalformat = GL_RGBA32F if c == 4 else GL_RGB32F
    glformat = GL_RGBA if c == 4 else GL_RGB
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, glinternalformat, h, w, imgCount)
    for i in range(imgCount):
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, h, w, 1, glformat, GL_FLOAT, _imgList[i])
    glBindTexture(GL_TEXTURE_3D, 0)
    return textureArray

#=======================================================

# download an image from the GPU
def downloadImage(texture, flipH=True, channels=4):
    glBindTexture(GL_TEXTURE_2D, texture)

    # PyOpenGL does not support downloading GL_RG textures
    # therefore: make it GL_RGB temporarily
    tempChannels = 3 if channels == 2 else channels
    
    _, glFormat = glFormatFromChannelCount(tempChannels)
    image = glGetTexImage(GL_TEXTURE_2D, 0, glFormat, GL_FLOAT)
    glBindTexture(GL_TEXTURE_2D, 0)
    if channels == 1:
        image = image[..., None]
    w, h, c = image.shape
    image = np.reshape(image, (h, w, c))

    # if temp channel was necessary, throw it away
    if not tempChannels == channels:
        image = image[..., 0:2]

    if flipH:
        image = np.flip(image, 0)
    return image
    
#=======================================================

# upload a mesh to the GPU
def uploadMesh(mesh, uploadTexture=True):
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    bufs = glGenBuffers(2)
    glBindBuffer(GL_ARRAY_BUFFER, bufs[0])
    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(ctypes.c_float) * len(mesh.verticesAndUVs),
        (ctypes.c_float * len(mesh.verticesAndUVs))(*mesh.verticesAndUVs),
        GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufs[1])
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        sizeof(ctypes.c_uint) * len(mesh.faces),
        (ctypes.c_uint * len(mesh.faces))(*mesh.faces),
        GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(
        0, 3, GL_FLOAT, GL_FALSE,
        5 * sizeof(ctypes.c_float), None)
    if uploadTexture:
        if mesh.texture is not None:
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(
                1, 2, GL_FLOAT, GL_FALSE,
                5 * sizeof(ctypes.c_float), ctypes.c_void_p(3 * sizeof(ctypes.c_float)))
            mesh.textureGPU = uploadImage(mesh.texture, flipH=False)
        else:
            print("No texture to upload.")
    else:
        mesh.textureGPU = None
    mesh.vao = vao
    mesh.gpu = True

#=======================================================

# upload a set of cameras to the GPU
def uploadCameras(cams, transposed):
    camBufferSize = 16+3+1  # (view-projection matrix, position, transposed)
    camDataFlattened = np.zeros((len(cams) * camBufferSize), dtype=np.float32)
    for i, cam in enumerate(cams):
        si = i * camBufferSize
        camDataFlattened[si:si+16] = np.transpose(cam.getFullTransform()).flatten()
        camDataFlattened[si+16:si+16+3] = cam.position.flatten()
        camDataFlattened[si+camBufferSize-1] = transposed[i]
    camBuffer = glGenBuffers(1)
    glBindBuffer(GL_UNIFORM_BUFFER, camBuffer)
    glBufferData(GL_UNIFORM_BUFFER, camDataFlattened, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_UNIFORM_BUFFER, 0)
    return camBuffer

#=======================================================

# render a screen-filling quad
def renderScreenQuad():
	glBegin(GL_QUADS)
	glTexCoord2f(0, 0)
	glVertex2f(-1, -1)
	glTexCoord2f(0, 1)
	glVertex2f(1, -1)
	glTexCoord2f(1, 1)
	glVertex2f(1, 1)
	glTexCoord2f(1, 0)
	glVertex2f(-1, 1)
	glEnd()

#=======================================================

# render quads from a vertex list
def renderQuads(vertices):
    glBegin(GL_QUADS)
    for v in vertices:
        glVertex3f(*v)
    glEnd()