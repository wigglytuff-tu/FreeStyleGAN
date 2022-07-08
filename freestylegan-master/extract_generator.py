# Extract generator from network pickle to avoid spending unnecessary memory

import sys, os
import pickle
import warnings

warnings.filterwarnings('ignore')

def getMainDir():
    return os.path.dirname(os.path.realpath(__file__))

def upDir(x):
   return os.path.dirname(x)

sys.path.append(upDir(getMainDir()))

import pretrained_networks

#----------------------------------------------------------------------------

def main(argv):
    ganFile = argv[1]
    G = pretrained_networks.load_networks(ganFile)[2]
    outputFile = os.path.join(getMainDir(), "data/networks/stylegan2_generator.pkl")
    pickle.dump(G, open(outputFile, "wb") )

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main(sys.argv)
    print("=== TERMINATED ===")
