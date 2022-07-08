# Script to convert a bundler camera path into lookat format

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
    inputRes = int(argv[2]), int(argv[3])

    outputCamFile = inputCamFile + ".lookat"

    cams = multiview.mvs_io.camerasFromBundler(inputCamFile, inputRes)
    
    multiview.mvs_io.camerasToLookat(outputCamFile, cams)

#=======================================================

if __name__ == "__main__":
    main(sys.argv)
    print("=== TERMINATED ===")