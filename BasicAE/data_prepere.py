import os
from aicsimageio import AICSImage
import numpy as np
import cv2

photo_limit = 1


def load(org_type):
    fovs = []
    folder = os.listdir("/storage/users/assafzar/fovs/" + org_type)
    counter = 0
    for t in folder:
        if ".tiff" in t and counter < photo_limit:
            fovs.append("/storage/users/assafzar/fovs/" + org_type + t)
            counter = counter + 1
    return fovs

    ## files, x_pixels, y_pixels, color


def separate_data(fovs, x, y, z):
    bright_field = []
    fluorescent = []
    for tiff in fovs:
        reader = AICSImage(tiff)
        img = reader.data

        img = np.squeeze(img, axis=0)
        n_channels = img.shape[0]

        mid_slice = np.int(0.5 * img.shape[1])
        img_2D = img[n_channels - 1, mid_slice, :, :]  ## [brighfiled, slice_z, all 2D image]
        img_2D_norm = (img_2D - np.min(img_2D)) / (np.max(img_2D) - np.min(img_2D))
        img_2D_color = np.expand_dims(img_2D_norm, axis=-1)  ## (624, 928) -> ((624, 928, 1))
        img_resized = cv2.resize(img_2D_color, (x, y))  ## (128, 128, 1)
        bright_field.append(img_resized)

        img_2D = img[n_channels - 2, mid_slice, :, :]
        img_2D_norm = (img_2D - np.min(img_2D)) / (np.max(img_2D) - np.min(img_2D))
        img_2D_color = np.expand_dims(img_2D_norm, axis=-1)  ## (624, 928) -> ((624, 928, 1))
        img_resized = cv2.resize(img_2D_color, (x, y))  ## (128, 128, 1)
        fluorescent.append(img_resized)

    bright_field_array = np.reshape(bright_field, (len(bright_field), x, y, z))
    fluorescent_array = np.reshape(fluorescent, (len(fluorescent), x, y, z))
    return bright_field_array, fluorescent_array
