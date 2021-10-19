import os
from aicsimageio import AICSImage
import numpy as np
import cv2



def load(org_type):
    fovs = []
    folder = os.listdir("/storage/users/assafzar/fovs/" + org_type)
    for t in folder:
        if ".tiff" in folder:
            fovs.append("/storage/users/assafzar/fovs/" + org_type + t)

    return fovs


def separate_data(fovs):
    bright_field = []
    fluorescent = []
    for tiff in fovs:
        reader = AICSImage(tiff)
        img = reader.data

        img = np.squeeze(img, axis=0)
        n_channels = img.shape[0]
        mid_slice = np.int(0.5 * img.shape[1])

        bright_field.append(cv2.resize(img[n_channels-1, mid_slice, :, :], (128, 128)))
        fluorescent.append(cv2.resize(img[n_channels-2, mid_slice, :, :], (128, 128)))
    return bright_field, fluorescent
