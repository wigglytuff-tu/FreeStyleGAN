import os
import numpy as np
from multiview.COLMAP.colmap_read_model import read_cameras_binary, read_images_binary

def parseColmapData(model_path, registration_path):
    # Load required files and folders
    sparse_path = os.path.join(model_path, "sparse")

    # Write bundler file
    write_bundler_file(sparse_path, registration_path)

    print("colmap data parsed successfully in Bundler file")


def parse_data(sparse_dir):
    camera_file = os.path.join(sparse_dir, 'cameras.bin')
    images_file = os.path.join(sparse_dir, 'images.bin')
    if not (os.path.exists(camera_file) and os.path.exists(images_file)):
        print("cameras.bin or images.bin not found")
        return

    # read binary files
    cameras = read_cameras_binary(camera_file) # Dictionary with only one entity as we used Shared intrinsics
    cameras = cameras[1]
    images = read_images_binary(images_file)
    num_camera = len(images)

    idx_sorted = np.argsort([images[key].name for key in images])  # sort data to be compatible with order of images
    idx_sorted = idx_sorted + 1  # because the keys in dictionary starts from 1

    focal, h, w, arr_R, arr_t = ([] for i in range(5))

    h = cameras.height
    w = cameras.width
    focal = cameras.params[0]
    for idx in idx_sorted:
        arr_R.append(images[idx].qvec2rotmat())
        arr_t.append(images[idx].tvec.reshape([3, 1]).squeeze())

    return num_camera, focal, h, w, arr_R, arr_t


def write_bundler_file(sparse_dir, registration_file):
    num_cams, focal, h, w, arr_R, arr_t = parse_data(sparse_dir)

    bundler_file = os.path.join(registration_file, 'bundle_0.out')

    with open(bundler_file, 'w') as f:
        f.write("# Bundle file v0.3\n")
        f.write(f"{num_cams}\n")
        for idx in range(num_cams):
            f.write(f"{int(focal)}\n")
            rot = arr_R[idx]
            for i in range(3):
                f.write(f"{rot[i][0]} {rot[i][1]} {rot[i][2]}\n")

            pos = arr_t[idx]
            f.write(f"{pos[0]} {pos[1]} {pos[2]}\n")