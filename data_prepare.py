import os
from aicsimageio import AICSImage
import numpy as np
import cv2
from enum import Enum

import utils


class ImgType(Enum):
    BRIGHT_FIELD = 0
    FLUORESCENT = 1


def load_paths(org_type, limit=1):
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
    z, y, x = img_size
    for tiff in fovs:
        reader = AICSImage(tiff)
        img = reader.data

        img = np.squeeze(img, axis=0)
        n_channels = img.shape[0]

        mid_slice = np.int(0.5 * img.shape[1])
        upper_slice = int(mid_slice + z / 2)
        under_slice = int(mid_slice - z / 2)
        img_3d = img[n_channels - 1, under_slice:upper_slice, :, :]  # [bright_field, slice_z, all 2D image]

        bright_field.append(image3d_prep(img_3d, ImgType.BRIGHT_FIELD))

        img_3d = img[n_channels - 4, under_slice:upper_slice, :, :]
        fluorescent.append(image3d_prep(img_3d, ImgType.FLUORESCENT))

    bright_field_array = np.array(bright_field)
    fluorescent_array = np.array(fluorescent)
    return bright_field_array, fluorescent_array


def image3d_prep(img_3d, type):
    z, y, x = img_3d.shape
    img_3d_padded = np.zeros((6, 640, 896))
    for i in range(z):
        img_3d_padded[i] = cv2.resize(img_3d[i], (896, 640), interpolation=cv2.INTER_AREA)
    img_3d_norm = utils.norm_img(img_3d_padded)
    return img_3d_norm
