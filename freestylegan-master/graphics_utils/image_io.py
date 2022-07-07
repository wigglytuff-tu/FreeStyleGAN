import os
import numpy as np
import cv2
import glob

#==============================================

def loadImage(path, normalize=True, appendAlpha=False):
    
    assert os.path.isfile(path), "Image file does not exist"
    
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if normalize:
        img = img.astype(np.float32) / 255.
    img[..., 0:3] = img[..., [2, 1, 0]]
    if appendAlpha and img.shape[2] == 3:
        alpha = np.ones_like(img[..., 0:1])
        img = np.concatenate([img, alpha], axis=-1)
    return img

#==============================================
    
def saveImage(img, path, channels=3):
    if img.ndim == 2:
        outImg = img[..., None]
    if img.ndim == 3 and img.shape[2] >= 2:
        if channels == 2:
            outImg = np.zeros((*img.shape[0:2], 3))
            outImg[..., 1:3] = img[..., 2::-1]
        if channels == 3:
            outImg = img[..., 2::-1]
        if channels == 4:
            outImg = img[..., [2, 1, 0, 3]]
    if outImg.dtype == np.float32 or outImg.dtype == np.float64:
        outImg = np.clip(outImg, 0, 1) * 255
        outImg = outImg.astype(np.uint8)
    if not cv2.imwrite(path, outImg):
        raise Exception("Could not write image! There should be something wrong with output path for imwrite!")

#==============================================

def findImages(path, ext=".png"):
    imgFiles = []
    for file in sorted(os.listdir(path)):
        if file.endswith(ext):
            imgFiles.append(os.path.join(path, file))   
    return imgFiles

#==============================================

def detectImageFormat(path):
    if len(glob.glob(os.path.join(path, "*.jpg"))) > 0:  # inconsistent behaviour of glob on Windows and Linux. glob is not case sensitive on windows. jpg and JPG should be double-checked!
        sample_case = glob.glob(os.path.join(path, "*.jpg"))[0]
        if sample_case.endswith(".jpg"):
            formatSuffix = ".jpg"
        elif sample_case.endswith(".JPG"):  # for windows
            formatSuffix = ".JPG"
    elif len(glob.glob(os.path.join(path, "*.JPG"))) > 0:  # for linux
        formatSuffix = ".JPG"
    elif len(glob.glob(os.path.join(path, "*.png"))) > 0:
        formatSuffix = ".png"
    else:
        assert False, "Check the format of images!"
    return formatSuffix