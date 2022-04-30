import json
import os
from pathlib import Path

from aicsimageio import AICSImage
import numpy as np
import cv2
from enum import Enum
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import shutil
import pandas as pd
import utils
from patchify import patchify, unpatchify

img_dims = (6, 640, 896)


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

def load_paths_v2(org_type, limit=None):  # Doesn't compel to pass organelle with '/'. limit default to max over min
    fovs = []
    storage_folder = '/' + os.path.join('storage', 'users', 'assafzar', 'fovs', org_type)
    file_paths = os.listdir(storage_folder)
    if limit is None:
        limit = len(file_paths)
    for file in file_paths:
        if file.endswith('.tiff'):
            fovs.append(os.path.join(storage_folder, file))
    return fovs[:limit]


def separate_data(fovs, img_size):
    bright_field = []
    fluorescent = []
    z, y, x = img_size
    for tiff in fovs:
        reader = AICSImage(tiff)
        img = reader.data

        img = np.squeeze(img, axis=0)
        n_channels = img.shape[0]
        if n_channels <= 6:
            continue
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


def predict_on_imgs(model, imgs, patch_dims):  # assuming img dims are (1,patch_dims)
    patches = utils.utils_patchify(imgs, patch_dims, over_lap_steps=1)
    for row in patches:
        for patch in row:
            patch[0] = model.predict(patch)[0]
    size = imgs[0].shape
    return utils.unpatchify(patches, size)


def predict_on_patches(model, patches: np.array, img_size_channels_last):  # assuming img dims are (1,patch_dims)
    fake_fluorescent_patches = []
    for patch in patches:
        patch = np.expand_dims(patch, axis=0)
        fake_fluorescent_patches.append(model.predict(patch))
    fake_fluorescent_patches = np.reshape(np.array(fake_fluorescent_patches), newshape=(5, 7, 1,) + patches[
        0].shape)  # (35,patch_size) -> ((5, 7, 1, patch_size)
    fake_fluorescent_img_channels_last = utils.unpatchify(fake_fluorescent_patches, img_size_channels_last)
    # fake_fluorescent_img_channels_first = np.moveaxis(fake_fluorescent_img_channels_last, -1, 0)
    return fake_fluorescent_img_channels_last
