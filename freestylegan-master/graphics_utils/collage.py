import sys, os
import image_io
import numpy as np
import math

#===================================================

def createCollage(files, numRows, verbose=False):

    imgCount = len(files)
    numCols = math.ceil(imgCount / numRows)
    if verbose:
        print("Found %i images. Creating %ix%i collage..." % (imgCount, numRows, numCols))

    res = image_io.loadImage(files[0]).shape
    comp = np.zeros((numRows * res[0], numCols * res[1], res[2]))

    for idx, f in enumerate(files):
        img = image_io.loadImage(f)
        assert img.shape == res, "Images must have same resolution."
        row = idx % numCols
        col = int(idx / numCols)
        comp[col*res[0]:(col+1)*res[0], row*res[1]:(row+1)*res[1], ::] = img
    
    return comp
    
#=======================================================

def main(argv):
    _, path, numRows = argv
    numRows = int(numRows)
    outputName = "collage.png"
    
    files = []

    for file in sorted(os.listdir(path)):
        if (file.endswith("png") or file.endswith("jpg")):
            if outputName in file:
                continue
            files.append(os.path.join(path, file))

    collage = createCollage(files, numRows, True)

    outPath = os.path.join(path, outputName)
    image_io.saveImage(collage, outPath)

#=======================================================

if __name__ == "__main__":    
    main(sys.argv)
    print("=== TERMINATED ===")