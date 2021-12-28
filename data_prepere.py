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
    return fovs[:min(limit, len(fovs))]

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
    pad_image = cv2.resize(img_2D, (896, 640), interpolation=cv2.INTER_AREA)
    # pad_image = cv2.copyMakeBorder(img_2D, 0, 16, 0, 100, cv2.BORDER_REPLICATE)
    img_2d_norm = utils.norm_img(pad_image)
    img_2d_color = np.expand_dims(img_2d_norm, axis=-1)  # (624, 928) -> ((624, 928, 1))
    return img_2d_color
