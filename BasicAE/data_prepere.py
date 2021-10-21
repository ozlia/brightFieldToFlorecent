import os
from aicsimageio import AICSImage
import numpy as np
import cv2



def load(org_type):
    fovs = []
    folder = os.listdir("/storage/users/assafzar/fovs/" + org_type)
    counter = 0
    for t in folder:
        if ".tiff" in t:
            fovs.append("/storage/users/assafzar/fovs/" + org_type + t)
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
        img_resized = img[n_channels-1, mid_slice, :, :] ## [brighfiled, slice_z, all 2D image]
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) ## (624, 928) -> ((624, 928, 3))
        img_resized = cv2.resize(img_resized, (x, y)) ## (128, 128, 3)
        bright_field.append(img_resized)
        img_resized = img[n_channels-2, mid_slice, :, :]
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_resized, (x, y))
        fluorescent.append(img_resized)

    bright_field_array = np.reshape(bright_field, (len(bright_field), x, y, z))
    fluorescent_array = np.reshape(fluorescent, (len(fluorescent), x, y, z))
    return bright_field_array, fluorescent_array
