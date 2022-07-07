import os
import numpy as np
import cv2
import dlib

def upDir(x):
   return os.path.dirname(x)

def getDir():
    return upDir(os.path.realpath(__file__))

#===============================================================

# initialize face detectors
# predictor file downloaded from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
def initDetectors():
    detector = dlib.get_frontal_face_detector()
    model_path = os.path.join(upDir(upDir(getDir())), "data", "networks", "shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor(model_path)
    return detector, predictor

#===============================================================

# detect 68 facial landmarks
def detectLandmarks(img, detector, predictor):

    NUM_FEATS = 68

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    facesBB = detector(gray)

    landmarks = np.zeros((NUM_FEATS, 2), dtype=np.int)

    success = len(facesBB) != 0

    if success:
        landmarksRaw = predictor(gray, facesBB[0])
        for i in range(NUM_FEATS):
            landmarks[i] = [landmarksRaw.part(i).x, landmarksRaw.part(i).y]
    
    return success, landmarks

#===============================================================

# aggregate landmarks to representative facial features
def findFeatures(landmarks):

    lm_eye_left      = landmarks[36 : 42]
    lm_eye_right     = landmarks[42 : 48]
    lm_mouth_outer   = landmarks[48 : 60]

    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    mouth        = np.mean(lm_mouth_outer, axis=0)

    return [eye_left, eye_right, mouth]
    
