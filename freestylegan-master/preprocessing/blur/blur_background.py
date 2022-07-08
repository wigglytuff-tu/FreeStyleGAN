import os, sys
import numpy as np
import warnings
import glob

import torch
import torchvision.transforms as transforms

from modnet import MODNet

def upDir(x):
   return os.path.dirname(x)

def getDir():
    return upDir(os.path.realpath(__file__))

sys.path.append(upDir(upDir(getDir())))

from graphics_utils.image_io import *

warnings.filterwarnings('ignore')

#=======================================================================

def inferMatte(im, modnet, im_transform, ref_size=512):

    # convert to PyTorch tensor
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    _, _, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = torch.nn.functional.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(im.cuda(), True)

    # resize matte
    matte = torch.nn.functional.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()

    return matte[..., None]

#=======================================================================

def overCompositing(fg, bg):
    return fg[..., 0:3] * fg[..., 3:] + bg[..., 0:3] * (1-fg[..., 3:])

#=======================================================================

def backgroundBlur(path, blurSigma=0.1, resizeFactor=0.25, saveMatte=True):

    dir_list = os.listdir(path)
    if "realitycapture" in dir_list:
        print("Run data from realitycapture")
        path = os.path.join(path, "realitycapture", "registration")
    elif "colmap" in dir_list:
        print("Run data from colmap")
        path = os.path.join(path, "colmap", "dense", "0", "images")
    else:
        assert False, "directory name cannot be matched!"

    model_path = os.path.join(upDir(upDir(getDir())), "data/networks/modnet_photographic_portrait_matting.ckpt")
    matte_str = "_matte"
    blur_str = "_blur"

    #------------------------------------------------------

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # load pretrained network
    modnet = MODNet(backbone_pretrained=False)
    modnet = torch.nn.DataParallel(modnet).cuda()
    modnet.load_state_dict(torch.load(model_path))
    modnet.eval()

    # find images
    formatSuffix = detectImageFormat(path)
    im_names = findImages(path, ext=formatSuffix)
    im_names = [img for img in im_names if matte_str not in img and blur_str not in img]
    im_names = [os.path.split(img)[1] for img in im_names]  # Separate the name of the image
    im_count = len(im_names)
    assert im_count > 0, "No images found"

    # processing
    for idx, im_name in enumerate(im_names):
        print('\rProcessing image %i/%i...' % (idx+1, im_count), end="")
        # load
        imFull = loadImage(os.path.join(path, im_name), appendAlpha=True)
        imRGB = imFull[..., 0:3]
        imAlpha = imFull[..., 3]
        res = imFull.shape[0:2][::-1]
       
        # inference
        matte = inferMatte(imRGB, modnet, im_transform)
        matteInv = 1 - matte

        # combine alpha channels
        fg = np.concatenate([imRGB, matte], axis=-1)
        bg = imRGB * matteInv
        bg = np.concatenate([bg, matteInv], axis=-1)
        bg[..., 3] = np.minimum(bg[..., 3], imAlpha)

        # blur background (resize first for speed)
        smallRes = (int(res[0] * resizeFactor), int(res[1] * resizeFactor))
        bg = cv2.resize(bg, smallRes, interpolation=cv2.INTER_AREA)
        bg = cv2.GaussianBlur(bg, (0, 0), blurSigma * bg.shape[1], borderType=cv2.BORDER_REFLECT)
        bg = np.where(bg[..., 3:] != 0, bg[..., :3] / bg[..., 3:], np.zeros_like(bg[..., :3]))
        bg = cv2.resize(bg, res, interpolation=cv2.INTER_AREA)

        # composite
        comp = overCompositing(fg, bg)

        # save
        img_name = im_name.split('.')[0]
        saveImage(comp, os.path.join(path, img_name + blur_str + '.png'), channels=3)
        if saveMatte:
            saveImage(matte[..., 0], os.path.join(path, img_name + matte_str + '.png'))

    print("")

#=======================================================================

def main(argv):
    path = argv[1]
    saveMatte = bool(int(argv[2]))
    backgroundBlur(path, saveMatte=saveMatte)

#=======================================================================

if __name__ == '__main__':
    main(sys.argv)
    print("=== TERMINATED ===")

   