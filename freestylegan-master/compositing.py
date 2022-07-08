# Tool to composite FreeStyleGAN renderings into synthetic environments

import os, sys

from graphics_utils.ogl_ops import *
from graphics_utils.render_utils import *
from graphics_utils.image_io import *
from graphics_utils.render_video import *
        
#==============================================

def main(argv):
    
    dataDir = argv[1]

    exportVideo = False  # needs ffmpeg

    # below this difference threshold, a boundary pixel is considered worth matching
    threshold = .25 

    # post-blur params
    postBlurSeam = True
    compBlur = 8
    maskBlur = 15
    boundaryBlurRegion = 10
    
    #--------------------------------------

    bgImgDir = os.path.join(dataDir, "bg/")
    fgImgDir = os.path.join(dataDir, "fg/")
    outputDir = os.path.join(dataDir, "comp/")

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    assert os.path.exists(bgImgDir) and os.path.exists(fgImgDir), "Need /fg and /bg folders containing images"

    print("Collecting images...") 
    bgImgFiles = findImages(bgImgDir)
    fgImgFiles = findImages(fgImgDir)
    assert len(bgImgFiles) == len(fgImgFiles), "Foreground and background image numbers don't match."
    imgCount = len(bgImgFiles)

    print("Initializing OGL...")
    res = 1024
    oglInit()
    fbo = createFramebuffer()
    
    morphologyOp = MorphologyOP(fbo, res)
    boundaryExtractionOp = BoundaryExtractionOP(fbo, res)
    convPyramidMembraneOp = ConvPyramidMembraneOP(fbo, res)
    gaussianBlurOp = GaussianBlurOP(fbo, res)

    for idx, (fgFile, bgFile) in enumerate(zip(fgImgFiles, bgImgFiles)):

        print("\rProcessing image %i/%i" % (idx+1, imgCount), end="")

        # load images
        bgImg = loadImage(bgFile)
        fgImg = loadImage(fgFile)
        bgTex = uploadImage(bgImg)
        fgTex = uploadImage(fgImg)

        # clean up foreground boundary by simply eroding
        fgTex = morphologyOp.render(fgTex, kernelSize=1, erode=True, alphaOnly=True)
        
        # find the boundary pixels
        boundaryMaskTex = boundaryExtractionOp.render(fgTex, bgTex, threshold) 
        
        # apply conv pyramids for membrane boundary interpolation
        compTex = convPyramidMembraneOp.render(fgTex, bgTex, boundaryMaskTex)
        compImg = downloadImage(compTex)

        # blur the seam
        if postBlurSeam:        
            
            # compute binary seam mask
            blurMaskTex = boundaryExtractionOp.render(fgTex, bgTex, 100) 
            blurMaskTex = morphologyOp.render(blurMaskTex, kernelSize=boundaryBlurRegion, erode=False, alphaOnly=False)
            
            # blur the mask
            blurMaskTex = gaussianBlurOp.render(blurMaskTex, maskBlur, (0,0,0,1))
            blurMask = downloadImage(blurMaskTex)[..., 3] 
            
            # blur the entire image
            compBlurTex = gaussianBlurOp.render(compTex, compBlur, (1,1,1,1))
            compBlurImg = downloadImage(compBlurTex)        
            
            # composite blurred image over original image, using the blurred mask as alpha
            compImg[..., 3] = 1 - blurMask
            compImg = overCompositing(compImg, compBlurImg)

        outFile = outputDir + "comp_" + str(idx).zfill(4)
        saveImage(compImg, outFile + ".png")
    
    print("")

    if exportVideo:
        renderVideo(outputDir, False)


#==============================================

if __name__ == "__main__":
    main(sys.argv)
    print("=== TERMINATED ===")