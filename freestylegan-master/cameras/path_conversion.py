# Script to convert a bundler camera path from/to canonical coordinates

import sys, os
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def getDir():
    return os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.dirname(getDir()))

import multiview.mvs_io

#=======================================================

def main(argv):
    
    inputCamFile = argv[1]
    modelMatrixFile = argv[2]
    outputCamFile = argv[3]
    resolution = int(argv[4]), int(argv[5])
    inverse = argv[6] != '0'

    modelMatrix = np.loadtxt(modelMatrixFile)

    nativeRes = (1024, 1024)

    inputRes = resolution if inverse else nativeRes
    outputRes = nativeRes if inverse else resolution

    inputModelMatrix = modelMatrix if inverse else None
    outputModelMatrix = None if inverse else np.linalg.inv(modelMatrix)

    cams = multiview.mvs_io.camerasFromBundler(inputCamFile, inputRes, inputModelMatrix)
    multiview.mvs_io.camerasToBundler(outputCamFile, cams, outputRes, outputModelMatrix)

#=======================================================

if __name__ == "__main__":
    main(sys.argv)
    print("=== TERMINATED ===")