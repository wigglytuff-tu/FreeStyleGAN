# PCA analysis of StyleGAN2 mapping network following https://arxiv.org/abs/2004.02546

import sys, os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
from sklearn.decomposition import IncrementalPCA

def getMainDir():
    return os.path.dirname(os.path.realpath(__file__))

def upDir(x):
   return os.path.dirname(x)

sys.path.append(upDir(getMainDir()))

import pretrained_networks

#==============================================

def main(argv):

    ganFile = argv[1]
    components = int(argv[2])
    seed = 100
    batchSize = components
    totalSamples = int(1e6)

    outputPath = os.path.join(getMainDir(), "data/pca/pca_" + str(components) + ".csv")

    #-------------------------------------
    
    M = pretrained_networks.load_networks(ganFile)[2].components.mapping
    
    rnd = np.random.RandomState(seed)
    ipca = IncrementalPCA(n_components=components)
    
    for it in range(int(totalSamples / batchSize)):
        if it % 10 == 0:
            print("\rit = %i/%i" % (it * batchSize, totalSamples), end='')
        z = rnd.randn(batchSize, *M.input_shape[1:])
        w = M.run(z, None)[::, 0, ::]
        ipca.partial_fit(w)

    print("")

    np.savetxt(outputPath, ipca.components_, delimiter=',')
    
#==============================================

if __name__ == "__main__":
    main(sys.argv)
    print("=== TERMINATED ===")