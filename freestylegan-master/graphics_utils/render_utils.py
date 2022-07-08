import os, sys
import numpy as np

def getUtilsDir():
    return os.path.dirname(os.path.realpath(__file__))

sys.path.append(getUtilsDir())

from cameras.camera import homogeneousTransform

#=======================================================

# CPU "over" compositing.
def overCompositing(fg, bg):
    return fg[..., 0:3] * fg[..., 3:] + bg[..., 0:3] * (1-fg[..., 3:])

#=======================================================

# Add an XY-plane to a mesh.
def addBackgroundPlane(mesh, depth=3., size=25., res=100):
    
    def createQuad(lowerLeft, upperRight):
        return np.array([
            [lowerLeft[0],  upperRight[1], -depth, 0, 0],
            [lowerLeft[0],  lowerLeft[1],  -depth, 0, 0],
            [upperRight[0], upperRight[1], -depth, 0, 0],
            [upperRight[0], upperRight[1], -depth, 0, 0],
            [lowerLeft[0],  lowerLeft[1],  -depth, 0, 0],
            [upperRight[0], lowerLeft[1],  -depth, 0, 0]], dtype=np.float32)    

    quadCount = res * res
    quadSize = 2 * size / res
    vertexCount = 6 * quadCount
    bgVertices = np.zeros((vertexCount, 5), dtype=np.float32)

    for y in range(res):
        for x in range(res):
            index = (y * res + x) * 6
            lowerLeft = (2 * (np.array([x, y]) / res) - 1) * size
            upperRight = lowerLeft + quadSize
            bgVertices[index:index+6] = createQuad(lowerLeft, upperRight)
    
    mesh.vertices = np.append(mesh.vertices, bgVertices[:, 0:3].flatten())
    mesh.verticesAndUVs = np.append(mesh.verticesAndUVs, bgVertices.flatten())
    mesh.faces = np.append(mesh.faces, np.arange(mesh.faces[-1]+1, mesh.faces[-1]+vertexCount+1))

#===============================================

# transform a mesh with a model matrix
def transformMesh(mesh, modelMatrix):
    verts = np.reshape(mesh.vertices, (int(mesh.vertices.shape[0] / 3), 3))
    for i, v in enumerate(verts):
        newV = homogeneousTransform(modelMatrix, v)
        mesh.vertices[3*i:3*(i+1)] = newV
        mesh.verticesAndUVs[5*i:5*i+3] = newV