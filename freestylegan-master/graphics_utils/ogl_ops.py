import os, sys
import math

def getUtilsDir():
    return os.path.dirname(os.path.realpath(__file__))

sys.path.append(getUtilsDir())

from ogl_utils import *

#=======================================================

# base class for OpenGL operations
class OpenGLOP:

    def __init__(
        self,
        fbo,
        vertexShaders, fragmentShaders, 
        renderTargetCount, 
        renderTargetResolution=None,
        createDepthRenderTargets=False):
        
        self.fbo = fbo

        # init shader(s)
        shaderDir = getUtilsDir() + "/shaders/"
        self.shaders = []
        if not isinstance(vertexShaders, list):
            vertexShaders = [vertexShaders]
        if not isinstance(fragmentShaders, list):
            fragmentShaders = [fragmentShaders]
        if len(vertexShaders) == 1 and len(fragmentShaders) > 1:
            vertexShaders = len(fragmentShaders) * vertexShaders
        for v, f in zip(vertexShaders, fragmentShaders):
            self.shaders.append(createShader(shaderDir + v, shaderDir + f))
        if len(self.shaders) == 1:
            self.shader = self.shaders[0]

        # init uniforms
        self.uniforms = EasyDict()
        
        # init render target(s)
        if renderTargetCount > 0:
            assert renderTargetResolution is not None, "Need resolution to create rendertarget."
            self.rtRes = renderTargetResolution
            if isinstance(self.rtRes, int):
                self.rtRes = (self.rtRes, self.rtRes)
            self.rendertargets = []
            for _ in range(renderTargetCount):
                self.rendertargets.append(createRenderTargets(self.rtRes, createDepthRenderTargets))

    def getUniform(self, name, shaderIdx=0):
        return glGetUniformLocation(self.shaders[shaderIdx], name)

#=======================================================

# Render a textured screen quad. Allows slicing of array textures and MIP levels and rendering directly to the screen
class TexturedScreenQuadOP(OpenGLOP):

    def __init__(self, fbo, resolution):
        OpenGLOP.__init__(self, fbo, "vertex2Duv", "textured_quad", 1, resolution)
        self.uniforms.outputRes = self.getUniform("outputRes")
        self.uniforms.level = self.getUniform("level")
        self.uniforms.layer = self.getUniform("layer")
        self.uniforms.transpose = self.getUniform("transposed")
        self.uniforms.remap = self.getUniform("remap")    

    def render(self, tex, toScreen, level=0, layer=-1, transpose=False, remap=False):
        
        if not toScreen:
            assert self.fbo, "Need FBO for rendering to a texture."
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.rendertargets[0].color, 0)

        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glViewport(0, 0, self.rtRes[0], self.rtRes[1])
        glUseProgram(self.shader)

        if layer == -1:
            target = GL_TEXTURE_2D 
            glActiveTexture(GL_TEXTURE0)
        else:
            target = GL_TEXTURE_2D_ARRAY
            glActiveTexture(GL_TEXTURE1)
        
        glBindTexture(target, tex)
        setTexParams(target)

        minFilter = GL_NEAREST if level == 0 else GL_NEAREST_MIPMAP_NEAREST
        glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minFilter)
        
        glProgramUniform2i(self.shader, self.uniforms.outputRes, self.rtRes[0], self.rtRes[1])
        glProgramUniform1i(self.shader, self.uniforms.level, level) 
        glProgramUniform1i(self.shader, self.uniforms.layer, layer) 
        glProgramUniform1i(self.shader, self.uniforms.transpose, transpose) 
        glProgramUniform1i(self.shader, self.uniforms.remap, remap) 
        glClear(GL_COLOR_BUFFER_BIT)
        renderScreenQuad()
        glBindTexture(target, 0)
        glUseProgram(0)

        if not toScreen:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            return self.rendertargets[0].color
        else:
            return None
    
#=======================================================

# g-Buffer rasterization. Supports position and depth buffer.
class GBufferOP(OpenGLOP):

    def __init__(self, fbo, resolution):
        OpenGLOP.__init__(self, fbo, "gbuffer", "gbuffer", 1, resolution, True)
        self.uniforms.modelviewMatrix = self.getUniform("modelviewMatrix")
        self.uniforms.projectionMatrix = self.getUniform("projectionMatrix")
        self.uniforms.camPosition = self.getUniform("camPosition")
        self.uniforms.mode = self.getUniform("mode")
        self.uniforms.lensDistortion = self.getUniform("lensDistortion")
        self.modes = {'Position' : 0, 'Depth' : 1 }

    def render(self, mesh, camera, resolution, mode='Position', indexedMesh=True, lensDistortion=(0,0)):
    
        assert mode in self.modes, "Mode should be in " + str(self.modes)

        if isinstance(resolution, int):
            resolution = (resolution, resolution)

        if self.rtRes != resolution:
            self.rendertargets[0] = createRenderTargets(resolution, True)
            self.rtRes = resolution
        
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.rendertargets[0].color)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.rendertargets[0].color, 0)
        glBindTexture(GL_TEXTURE_2D, self.rendertargets[0].depth)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.rendertargets[0].depth, 0)
        
        glUseProgram(self.shader)
        glEnable(GL_DEPTH_TEST)
        glCullFace(GL_BACK)
        glEnable(GL_CULL_FACE)
        glViewport(0, 0, self.rtRes[0], self.rtRes[1])
        
        if indexedMesh:
            glBindVertexArray(mesh.vao)
  
        glProgramUniformMatrix4fv(self.shader, self.uniforms.modelviewMatrix, 1, True, camera.viewMatrix)
        glProgramUniformMatrix4fv(self.shader, self.uniforms.projectionMatrix, 1, True, camera.projectionMatrix)
        glProgramUniform3f(self.shader, self.uniforms.camPosition, *camera.position)
        glProgramUniform1i(self.shader, self.uniforms.mode, self.modes[mode])
        glProgramUniform2f(self.shader, self.uniforms.lensDistortion, *lensDistortion)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if indexedMesh:
            glDrawElements(GL_TRIANGLES, len(mesh.faces), GL_UNSIGNED_INT, None)  
        else:
            renderQuads(mesh)
        
        glBindVertexArray(0)
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        return self.rendertargets[0].color
        
#=======================================================

# Textured mesh rasterization.
class TexturedMeshOP(OpenGLOP):

    def __init__(self, fbo, resolution):
        OpenGLOP.__init__(self, fbo, "textured_mesh", "textured_mesh", 1, resolution, True)
        self.uniforms.viewMatrix = self.getUniform("modelviewMatrix")
        self.uniforms.projectionMatrix = self.getUniform("projectionMatrix")
        self.uniforms.lensDistortion = self.getUniform("lensDistortion")
        
    def render(self, mesh, camera, lensDistortion=(0,0)):
                
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.rendertargets[0].color)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.rendertargets[0].color, 0)
        glBindTexture(GL_TEXTURE_2D, self.rendertargets[0].depth)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.rendertargets[0].depth, 0)

        glUseProgram(self.shader)
        glEnable(GL_DEPTH_TEST)
        glCullFace(GL_BACK)
        glEnable(GL_CULL_FACE)
        glViewport(0, 0, self.rtRes[0], self.rtRes[1])
        
        glBindVertexArray(mesh.vao)  

        glProgramUniformMatrix4fv(self.shader, self.uniforms.viewMatrix, 1, True, camera.viewMatrix)
        glProgramUniformMatrix4fv(self.shader, self.uniforms.projectionMatrix, 1, True, camera.projectionMatrix)
        glProgramUniform2f(self.shader, self.uniforms.lensDistortion, *lensDistortion)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, mesh.textureGPU)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawElements(GL_TRIANGLES, len(mesh.faces), GL_UNSIGNED_INT, None)  
        
        glUseProgram(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindVertexArray(0)
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)

        return self.rendertargets[0].color
        
#==============================================

# Unstructured lumigraph rendering.
# http://cs.harvard.edu/~sjg/papers/ulr.pdf
class ULROP(OpenGLOP):

    def __init__(self, fbo, resolution):
        OpenGLOP.__init__(self, fbo, "vertex2Duv", "ulr", 1, resolution, False)
        self.uniforms.novelCamPosition = self.getUniform("novelCamPosition")
        self.uniforms.camsCount = self.getUniform("camsCount")
        self.uniforms.showBlendingWeights = self.getUniform("showBlendingWeights")
        self.uniforms.varianceWeightToAlpha = self.getUniform("varianceWeightToAlpha")
        self.uniforms.varianceWeightToRGB = self.getUniform("varianceWeightToRGB")

    def render(
        self,
        gBuffer, camPosition,
        origImgs, origGBuffers, origCams,
        showBlendingWeights=False, varianceWeightToAlpha=False, varianceWeightToRGB=False):
        
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.rendertargets[0].color, 0)

        glViewport(0, 0, self.rtRes[0], self.rtRes[1])

        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.shader)

        glBindTexture(GL_TEXTURE_2D_ARRAY, origImgs)
        camCount = glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_DEPTH)
        
        glProgramUniform3f(self.shader, self.uniforms.novelCamPosition, *(camPosition))
        glProgramUniform1i(self.shader, self.uniforms.camsCount, camCount)
        glProgramUniform1i(self.shader, self.uniforms.showBlendingWeights, showBlendingWeights)
        glProgramUniform1i(self.shader, self.uniforms.varianceWeightToAlpha, varianceWeightToAlpha)
        glProgramUniform1i(self.shader, self.uniforms.varianceWeightToRGB, varianceWeightToRGB)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, gBuffer)
        setTexParams()

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, origImgs)
        setTexParams(GL_TEXTURE_2D_ARRAY)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D_ARRAY, origGBuffers)
        setTexParams(GL_TEXTURE_2D_ARRAY)

        glBindBuffer(GL_UNIFORM_BUFFER, origCams)
        glBindBufferBase(GL_UNIFORM_BUFFER, 3, origCams)

        glClear(GL_COLOR_BUFFER_BIT)
        renderScreenQuad()
        glActiveTexture(GL_TEXTURE0)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        return self.rendertargets[0].color
        
#==============================================

# Simple morphological filter: Erosion and dilation.
class MorphologyOP(OpenGLOP):

    def __init__(self, fbo, resolution):
        OpenGLOP.__init__(self, fbo, "vertex2Duv", "morphology", 1, resolution, False)
        self.uniforms.kernelSize = self.getUniform("kernelSize")
        self.uniforms.erode = self.getUniform("erode")
        self.uniforms.alphaOnly = self.getUniform("alphaOnly")
        
    def render(self, inputTexture, kernelSize, erode, alphaOnly):

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.rendertargets[0].color, 0)

        glViewport(0, 0, self.rtRes[0], self.rtRes[1])

        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)

        glUseProgram(self.shader)
        glProgramUniform1i(self.shader, self.uniforms.kernelSize, kernelSize)
        glProgramUniform1i(self.shader, self.uniforms.erode, erode)
        glProgramUniform1i(self.shader, self.uniforms.alphaOnly, alphaOnly)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, inputTexture)
        setTexParams()
        
        glClear(GL_COLOR_BUFFER_BIT)
        renderScreenQuad()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        return self.rendertargets[0].color

#==============================================

# Two textures to one anaglyph texture
class AnaglyphOP(OpenGLOP):

    def __init__(self, fbo, resolution):
        OpenGLOP.__init__(self, fbo, "vertex2Duv", "anaglyph", 1, resolution, False)


    def render(self, leftTexture, rightTexture):
         
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.rendertargets[0].color, 0)

        glViewport(0, 0, self.rtRes[0], self.rtRes[1])

        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)

        glUseProgram(self.shader)
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, leftTexture)
        setTexParams()

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, rightTexture)
        setTexParams()
        
        glClear(GL_COLOR_BUFFER_BIT)
        renderScreenQuad()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        return self.rendertargets[0].color
        
#==============================================

# Rasterize geometry using camera1 and "shade" with lookup position from camera0.
class FlowOP(OpenGLOP):
    
    def __init__(self, fbo, resolution, rendertargetCount=1):
        OpenGLOP.__init__(self, fbo, "flow", "flow", rendertargetCount, resolution, True)
        self.uniforms.viewMatrix0 = self.getUniform("viewMatrix0")
        self.uniforms.viewMatrix1 = self.getUniform("viewMatrix1")
        self.uniforms.projectionMatrix0 = self.getUniform("projectionMatrix0")
        self.uniforms.projectionMatrix1 = self.getUniform("projectionMatrix1")
        self.uniforms.lensDistortion = self.getUniform("lensDistortion")
        self.uniforms.zClamp = self.getUniform("zClamp")

    def render(self, mesh, camera0, camera1, rtID=0, lensDistortion=(0,0), zClamp=1000):
        glUseProgram(self.shader)
        glEnable(GL_DEPTH_TEST)
        glCullFace(GL_BACK)
        glEnable(GL_CULL_FACE)
        glViewport(0, 0, self.rtRes[0], self.rtRes[1])
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glBindVertexArray(mesh.vao)
      
        glProgramUniformMatrix4fv(self.shader, self.uniforms.viewMatrix0, 1, True, camera0.viewMatrix)
        glProgramUniformMatrix4fv(self.shader, self.uniforms.projectionMatrix0, 1, True, camera0.projectionMatrix)
        glProgramUniform1f(self.shader, self.uniforms.zClamp, zClamp)
        glProgramUniformMatrix4fv(self.shader, self.uniforms.viewMatrix1, 1, True, camera1.viewMatrix)
        glProgramUniformMatrix4fv(self.shader, self.uniforms.projectionMatrix1, 1, True, camera1.projectionMatrix)
        glProgramUniform2f(self.shader, self.uniforms.lensDistortion, *lensDistortion)
    
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.rendertargets[rtID].color, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.rendertargets[rtID].depth, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawElements(GL_TRIANGLES, len(mesh.faces), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)
        glActiveTexture(GL_TEXTURE0)
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

        return self.rendertargets[rtID].color

#=======================================================

# Warping of a texture using a lookup coordinate per pixel.
class WarpingOP(OpenGLOP):

    def __init__(self, fbo, resolution):
        OpenGLOP.__init__(self, fbo, "vertex2Duv", "warping", 1, resolution)
    
    def render(self, inputTexture, flowTexture):

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.rtRes[0], self.rtRes[1])
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.shader)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, inputTexture)
        setTexParams(minFilter=GL_LINEAR)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, flowTexture)
        setTexParams()
            
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.rendertargets[0].color, 0)
        renderScreenQuad()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)

        return self.rendertargets[0].color    

#=======================================================

# Supersampling of a texture by reducing the resolution
class SupersamplingOP(OpenGLOP):

    def __init__(self, fbo, resolution, rendertargetCount=1):
        OpenGLOP.__init__(self, fbo, "vertex2Duv", "supersampling", rendertargetCount, resolution)
        self.uniforms.factor = self.getUniform("factor")

    def render(self, inputTexture, factor, rtID=0):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.rtRes[0], self.rtRes[1])
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.shader)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, inputTexture)
        setTexParams()

        glProgramUniform1i(self.shader, self.uniforms.factor, factor)
        
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.rendertargets[rtID].color, 0)
        renderScreenQuad()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)

        return self.rendertargets[rtID].color

#=======================================================

# Extract foreground boundaries, used for membrane interpolation (below).
class BoundaryExtractionOP(OpenGLOP):
    
    def __init__(self, fbo, resolution):
        OpenGLOP.__init__(self, fbo, "vertex2Duv", "boundaryExtraction", 1, resolution)
        self.uniforms.threshold = self.getUniform("threshold")

    def render(self, fgTexture, bgTexture, threshold):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.rtRes[0], self.rtRes[1])
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.shader)

        glProgramUniform1f(self.shader, self.uniforms.threshold, threshold)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, fgTexture)
        setTexParams()
        
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, bgTexture)
        setTexParams()

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.rendertargets[0].color, 0)
        renderScreenQuad()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)

        return self.rendertargets[0].color

#==============================================

# Implementation of Sec. 5 of https://www.cs.huji.ac.il/labs/cglab/projects/convpyr/data/convpyr-small.pdf
class ConvPyramidMembraneOP(OpenGLOP):

    def __init__(self, fbo, resolution):
        OpenGLOP.__init__(
            self, fbo, 
            "vertex2Duv", ["convPyramidMembraneAnalysis", "convPyramidMembraneSynthesis"], 
            2, resolution)
        self.uniforms.analysisLevel = self.getUniform("level", 0)
        self.uniforms.synthesisLevel = self.getUniform("level", 1)
        self.uniforms.synthesisMaxLevel = self.getUniform("maxLevel", 1)
        for rt in self.rendertargets:
            buildMIP(rt.color)

    def render(self, fgTexture, bgTexture, boundaryTexture):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.rtRes[0], self.rtRes[1])
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        
        levelCount = int(math.log(max(self.rtRes[0], self.rtRes[1]), 2)) + 1
        glProgramUniform1i(self.shaders[1], self.uniforms.synthesisMaxLevel, levelCount - 1)
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, boundaryTexture)
        setTexParams()
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, fgTexture)
        setTexParams()
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, bgTexture)
        setTexParams()
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, self.rendertargets[0].color)
        setTexParams(target=GL_TEXTURE_2D, minFilter=GL_NEAREST_MIPMAP_NEAREST, magFilter=GL_NEAREST, wrap=GL_CLAMP_TO_EDGE)
        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_2D, self.rendertargets[1].color)
        setTexParams(target=GL_TEXTURE_2D, minFilter=GL_NEAREST_MIPMAP_NEAREST, magFilter=GL_NEAREST, wrap=GL_CLAMP_TO_EDGE)

        # analysis
        glUseProgram(self.shaders[0])
        for l in range(0, levelCount):
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.rendertargets[0].color, l)
            glProgramUniform1i(self.shaders[0], self.uniforms.analysisLevel, l)
            renderScreenQuad()

        # synthesis
        glUseProgram(self.shaders[1])
        for l in range(levelCount-1, -1, -1):
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.rendertargets[1].color, l)
            glProgramUniform1i(self.shaders[1], self.uniforms.synthesisLevel, l)
            renderScreenQuad()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)

        return self.rendertargets[1].color

#=======================================================

# Gaussian blur
class GaussianBlurOP(OpenGLOP):

    def __init__(self, fbo, resolution):
        OpenGLOP.__init__(self, fbo, "vertex2Duv", "gaussian_filter", 1, resolution)
        self.uniforms.radius = self.getUniform("radius")
        self.uniforms.channels = self.getUniform("channels")

    def render(self, texture, radius, channels):

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.rtRes[0], self.rtRes[1])
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.shader)
        glProgramUniform1i(self.shader, self.uniforms.radius, radius)
        glProgramUniform4i(self.shader, self.uniforms.channels, *channels)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)
        setTexParams()

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.rendertargets[0].color, 0)
        renderScreenQuad()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)

        return self.rendertargets[0].color