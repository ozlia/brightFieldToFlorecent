import os
from aicsimageio import AICSImage
import numpy as np
from patchify import patchify, unpatchify


def load(org_type, limit=1):
    fovs = []
    folder = os.listdir("/storage/users/assafzar/fovs/" + org_type)
    for t in folder:
        if ".tiff" in t:
            fovs.append("/storage/users/assafzar/fovs/" + org_type + t)
    if limit > len(fovs):
        limit = len(fovs) - 1
    return fovs[:limit]

    # files, x_pixels, y_pixels, color


def separate_data(fovs, img_size):
    bright_field = []
    fluorescent = []
    x, y, z = img_size
    for tiff in fovs:
        reader = AICSImage(tiff)
        img = reader.data

        img = np.squeeze(img, axis=0)
        n_channels = img.shape[0]

        mid_slice = np.int(0.5 * img.shape[1])
        img_2D = img[n_channels - 1, mid_slice, :, :]  # [bright_field, slice_z, all 2D image]
        bright_field.extend(image2d_prep(img_2D, x, y))

        img_2D = img[n_channels - 2, mid_slice, :, :]
        fluorescent.extend(image2d_prep(img_2D, x, y))

    bright_field_array = np.reshape(bright_field, (len(bright_field), x, y, z))
    fluorescent_array = np.reshape(fluorescent, (len(fluorescent), x, y, z))
    return bright_field_array, fluorescent_array


def image2d_prep(img_2D, x, y):
    img_2D_norm = (img_2D - np.min(img_2D)) / (np.max(img_2D) - np.min(img_2D))
    img_2D_color = np.expand_dims(img_2D_norm, axis=-1)  # (624, 928) -> ((624, 928, 1))
    patches = patchify(img_2D_color, (x, y, 1), step=x)  # split image into 35  128*128 patches. (4, 7, 1, 128, 128, 1)

    return resize_patch_list(patches) # resizes the patch such that (4, 7, 1, 128, 128, 1) -> (28, 128,128,1)


def resize_patch_list(patches):  # return shape of (28, 128,128,1)
    patches_list4D = []
    for i in patches:
        for j in i:
            patches_list4D.append(j[0, :, :, :])
    return patches_list4D
