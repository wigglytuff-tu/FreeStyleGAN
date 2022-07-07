import numpy as np
import math
import copy
from scipy.spatial.transform import Rotation

#=======================================================

# perspective camera model
class Camera:

    def __init__(self):
        self.viewMatrix = np.eye(4)
        self.setProjectionParams(fov=0.1)
        self.aspect = 1
        self.position = np.array([0, 0, 0])

    def setViewMatrix(self, v):
        self.viewMatrix = v
        self.position = camPositionFromViewMatrix(v)
    
    def setProjectionMatrix(self, p):
        self.projectionMatrix = p
        self.aspect = p[1,1] / p[0,0]
        self.fov = 2 * math.atan(1 / p[1,1])

    def setProjectionParams(self, fov, aspect=1):
        self.projectionMatrix = getProjectionMatrix(fov, aspect)
        self.fov = fov
        self.aspect = aspect

    def getFullTransform(self):
        return self.projectionMatrix.dot(self.viewMatrix)

#=======================================================

# create a 3D rotation matrix from three angles
def getRotationMatrix(angles, center=None, order='yxz'):
    m = np.eye(4)
    m[0:3, 0:3] = Rotation.from_euler(order, angles, degrees=True).as_matrix()
    if center is not None:
        shift1 = getTranslationMatrix(-center)
        shift2 = getTranslationMatrix(center)
        m = shift2.dot(m).dot(shift1)
    return m

#=======================================================

# create a 3D rotation matrix from an axis and an angle
# https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
def getRotationMatrixAxisAngle(axis, angle, center=None):
    a = np.radians(angle)
    c = np.cos(a)
    cm = 1 - c
    s = np.sin(a)
    r = np.eye(4, dtype=np.float32)
    r[0,0] = c + axis[0] * axis[0] * cm
    r[0,1] = axis[0] * axis[1] * cm - axis[2] * s
    r[0,2] = axis[0] * axis[2] * cm + axis[1] * s
    r[1,0] = axis[1] * axis[0] * cm + axis[2] * s
    r[1,1] = c + axis[1] * axis[1] * cm
    r[1,2] = axis[1] * axis[2] * cm - axis[0] * s
    r[2,0] = axis[2] * axis[0] * cm - axis[1] * s
    r[2,1] = axis[2] * axis[1] * cm + axis[0] * s
    r[2,2] = c + axis[2] * axis[2] * cm
    if center is not None:
        shift1 = getTranslationMatrix(-center)
        shift2 = getTranslationMatrix(center)
        r = shift2.dot(r).dot(shift1)
    return r

#=======================================================

# create a 3D translation matrix
def getTranslationMatrix(translation):
    m = np.eye(4)
    m[0:3, 3] = translation
    return m

#=======================================================

# create a 3D isotropic scaling matrix
def getIsotropicScalingMatrix(s):
    m = np.eye(4)
    m[0:3, 0:3] *= s
    return m

#=======================================================

# create a 3D anisotropic scaling matrix
def getAnisotropicScalingMatrix(s):
    m = np.eye(4)
    m[0,0] = s[0]
    m[1,1] = s[1]
    m[2,2] = s[2]
    return m

#=======================================================

# create a view matrix using position, look-at vector and up vector
def getLookAtViewMatrix(pos, at, up):
        z = normalize(pos - at)
        x = normalize(np.cross(normalize(up), z))
        y = normalize(np.cross(z, x))
        rot = np.transpose(np.stack([x, y, z]))
        r = np.eye(4)
        r[0:3, 0:3] = rot
        t = getTranslationMatrix(pos)
        return np.linalg.inv(t.dot(r))

#=======================================================

# create a perspective projection matrix
def getProjectionMatrix(fovy, aspect=1, znear=1, zfar=1000):
    scale = 1. / math.tan(0.5 * fovy)
    proj = np.zeros((4,4))
    proj[0, 0] = scale / aspect
    proj[1, 1] = scale
    proj[3, 2] = -1
    proj[2, 2] = - (zfar + znear) / (zfar - znear)
    proj[2, 3] = - 2 * zfar * znear / (zfar - znear)
    return proj

#=======================================================

# obtain camera position from view matrix
def camPositionFromViewMatrix(v):
    return np.linalg.inv(v)[0:3, 3]

#=======================================================

# obtain view direction from view matrix
def viewDirectionFromViewMatrix(v):
    invRot = np.linalg.inv(v[0:3, 0:3])
    return invRot.dot(np.array([0, 0, -1]))

#=======================================================

# obtain up vector from view matrix
def upVectorFromViewMatrix(v):
    invRot = np.linalg.inv(v[0:3, 0:3])
    return invRot.dot(np.array([0, 1, 0]))

#=======================================================

# convert field of view to focal length
def fovToFocalLength(fov, res):
    return res / (2 * math.tan(0.5 * fov))

#=======================================================

# convert focal length to field of view
def focalLengthToFov(focalLength, res):
    return 2 * math.atan(0.5 * res / focalLength)

#=======================================================

# transform a camera using a model matrix
def transformCamera(camera, modelMatrix):
    cam = copy.deepcopy(camera)
    pos = cam.position
    r = np.transpose(cam.viewMatrix[0:3, 0:3])
    center = pos + r.dot(np.array([0, 0, -1]))
    up = pos + r.dot(np.array([0, 1, 0]))
    pos = homogeneousTransform(modelMatrix, pos)
    center = homogeneousTransform(modelMatrix, center)
    up = homogeneousTransform(modelMatrix, up)
    viewMatrix = getLookAtViewMatrix(pos, center, normalize(up - pos))
    cam.setViewMatrix(viewMatrix)
    return cam
    
#===============================================

# create toe-in stereo matrices from a mono camera
def stereoFromMonoCam(camera, eyeSeparation=2, shiftScreen=0.05, res=(1024, 1024)):

    def frustumProjection(left, right, top, bottom, near, far):
        p = np.zeros((4,4), np.float32)        
        p[0,0] = 2 * near / (right - left)
        p[0,2] = (right + left) / (right - left)
        p[1,1] = (2 * near / (top - bottom))
        p[1,2] = (top + bottom) / (top - bottom)
        p[2,2] = (near + far) / (near - far)
        p[2,3] = 2 * near * far / (near - far)
        p[3,2] = -1
        return p

    leftEye = copy.deepcopy(camera)
    rightEye = copy.deepcopy(camera)

    # view matrices
    tVec = np.array([0.5 * eyeSeparation, 0, 0])
    pos = camera.position
    r = np.transpose(camera.viewMatrix[0:3, 0:3])
    up = r.dot(np.array([0, 1, 0]))
    at = pos + r.dot(np.array([0, 0, -1]))
    leftView = getLookAtViewMatrix(pos - tVec, at - tVec * (1 - shiftScreen), up)
    leftEye.setViewMatrix(leftView)
    rightView = getLookAtViewMatrix(pos + tVec, at + tVec * (1 - shiftScreen), up)
    rightEye.setViewMatrix(rightView)

    # projection matrices
    near = 1
    far = 1000
    focalLength = fovToFocalLength(camera.fov, res[1])
    wd2 = near * math.tan(0.5 * camera.fov) 
    ndfl = near / focalLength
    leftProj = frustumProjection(
        -camera.aspect * wd2 + 0.5 * eyeSeparation * ndfl,
        camera.aspect * wd2 + 0.5 * eyeSeparation * ndfl,
        wd2, -wd2, 
        near, far)
    rightProj = frustumProjection(
        -camera.aspect * wd2 - 0.5 * eyeSeparation * ndfl,
        camera.aspect * wd2 - 0.5 * eyeSeparation * ndfl,
        wd2, -wd2, 
        near, far)           
    leftEye.setProjectionMatrix(leftProj)
    rightEye.setProjectionMatrix(rightProj)

    return leftEye, rightEye

#=======================================================

# normalize a vector
def normalize(x):
    return x / np.linalg.norm(x)

#=======================================================

# perform perspective division
def perspDiv2D(x):
    return x[0:2] / x[3]

#=======================================================

# apply a homogeneous transform to a point
def homogeneousTransform(matrix, p):
    return matrix.dot(np.array([*p, 1]))[0:3]