import os
from aicsimageio import AICSImage
import numpy as np
from patchify import patchify, unpatchify
import cv2

import utils


def load(org_type, limit=1):
    fovs = []
    folder = os.listdir("/storage/users/assafzar/fovs/" + org_type)
    for t in folder:
        if ".tiff" in t:
            fovs.append("/storage/users/assafzar/fovs/" + org_type + t)

    return fovs[:min(limit, len(fovs) - 1)]

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

        bright_field.append(image2d_prep(img_2D, x, y))

        img_2D = img[n_channels - 4, mid_slice, :, :]
        fluorescent.append(image2d_prep(img_2D, x, y))

    bright_field_array = np.array(bright_field)
    fluorescent_array = np.array(fluorescent)
    return bright_field_array, fluorescent_array


def image2d_prep(img_2D, x, y, z=1):

    pad_image = cv2.copyMakeBorder(img_2D, 0, 16, 0, 100, cv2.BORDER_REPLICATE)

    img_2D_norm = (pad_image - np.min(pad_image)) / (np.max(pad_image) - np.min(pad_image))
    img_2D_color = np.expand_dims(img_2D_norm, axis=-1)  # (624, 928) -> ((624, 928, 1))

    return img_2D_color # resizes the patch such that (4, 7, 1, 128, 128, 1) -> (28, 128,128,1)


def resize_patch_list(patches):  # return shape of (28, 128,128,1)
    patches_list4D = []
    for i in patches:
        for j in i:
            patches_list4D.append(j[0, :, :, :])
    return patches_list4D


def pack_unpack_data(tiff, img_size):

    x, y, z = img_size
    reader = AICSImage(tiff)
    img = reader.data
    img = np.squeeze(img, axis=0)
    n_channels = img.shape[0]
    mid_slice = np.int(0.5 * img.shape[1])
    img_2D = img[n_channels - 1, mid_slice, :, :]  # [bright_field, slice_z, all 2D image]

    bright_field = do_patches_for_predict(img_2D, x, y)

    # img_2D = img[n_channels - 4, mid_slice, :, :]
    # fluorescent = do_patches_for_predict(img_2D, x, y)

    return bright_field


def do_patches_for_predict(img_2D, x, y, z=1):

    pad_image = cv2.copyMakeBorder(img_2D, 0, 16, 0, 100, cv2.BORDER_REPLICATE)
    img_2D_norm = (pad_image - np.min(pad_image)) / (np.max(pad_image) - np.min(pad_image))
    img_2D_color = np.expand_dims(img_2D_norm, axis=-1)  # (624, 928) -> ((624, 928, 1))
    # utils.save_full_2d_pic(img_2D_color)
    patches = patchify(img_2D_color, (x, y, z), step=x)  # split image into 35  128*128 patches. (4, 7, 1, 128, 128, 1)

    return patches