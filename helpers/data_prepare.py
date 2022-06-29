import os
from aicsimageio import AICSImage
import numpy as np
import cv2
from enum import Enum
from datetime import datetime, timedelta
import utils
import pandas as pd

TIFF_DF = pd.DataFrame()


class ImgType(Enum):
    BRIGHT_FIELD = 0
    FLUORESCENT = 1


def load_paths(org_type, limit=1):

    fovs = []
    if not org_type[-1] == "/":
        org_type = org_type + "/"
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


def separate_data(fovs, img_size, multiply_img_z=1, slice_by_brightest=False):
    bright_field = []
    fluorescent = []
    z, y, x = img_size
    tiff_to_read = len(fovs)
    print("Now Reading %d tiffs from disk, should take about %s (hr-min-sec)" % (
        tiff_to_read, timedelta(seconds=20 * tiff_to_read)))
    for i, tiff in enumerate(fovs):
        print("reading tiff number %d out of %d" % ((i + 1), tiff_to_read))
        start = datetime.now()
        reader = AICSImage(tiff)
        img = reader.data

        img = np.squeeze(img, axis=0)
        n_channels = img.shape[0]
        if n_channels <= 6:
            continue
        if slice_by_brightest:
            prep_img = lambda single_brightfield_or_floro: slice_img(image3d_prep(single_brightfield_or_floro))
            bright_field.append(prep_img(img[n_channels - 1, :, :, :]))
            fluorescent.append(prep_img(img[n_channels - 4, :, :, :]))
        else:
            mid_slice = np.int(0.5 * img.shape[1])
            under_slice = int(mid_slice - z * multiply_img_z / 2)
            for i in range(multiply_img_z):
                img_3d = img[n_channels - 1, under_slice:under_slice + z, :, :]  # [bright_field, slice_z, all 2D image]

                bright_field.append(image3d_prep(img_3d, ImgType.BRIGHT_FIELD))

                img_3d = img[n_channels - 4, under_slice:under_slice + z, :, :]
                fluorescent.append(image3d_prep(img_3d, ImgType.FLUORESCENT))
                under_slice += z
        stop = datetime.now()
        print('Done, Time for this tiff: ', stop - start)
    bright_field_array = np.array(bright_field)
    fluorescent_array = np.array(fluorescent)
    return bright_field_array, fluorescent_array


def image3d_prep(img_3d):
    z, y, x = img_3d.shape
    img_3d_padded = np.zeros((z, 640, 896))
    for i in range(z):
        img_3d_padded[i] = cv2.resize(img_3d[i], (896, 640), interpolation=cv2.INTER_AREA)
    img_3d_norm = utils.norm_img(img_3d_padded)
    return img_3d_norm


def get_brightest_layer(img_3d_channels_first) -> int:
    '''
    @param img_3d:
    @return: int of z layer with brightest pixels (by mean only)
    '''
    brightest_z_layer = -1
    max_mean = -1
    z_layers = img_3d_channels_first.shape[0]
    for curr_z_layer in range(z_layers):
        curr_slice_pixel_wise_mean = np.mean(img_3d_channels_first[curr_z_layer])
        if curr_slice_pixel_wise_mean > max_mean:
            max_mean = curr_slice_pixel_wise_mean
            brightest_z_layer = curr_z_layer
    return brightest_z_layer


def slice_img(img_3d_channels_first, num_z_slices=32):
    '''

    @param img_3d_channels_first:
    @param chosen_z_layer:
    @return: chooses upto 32 slices out of each direction (up and down) from the chosen slice. if not available, chooses randomly.
    '''
    assert len(img_3d_channels_first.shape) == 3, 'expecting 3d img with channels first'
    assert num_z_slices % 2 == 0, 'expecting an attempt to take equal number of slices from above and below the chosen z slice, so total number of z slices must be even'
    chosen_z_layer = get_brightest_layer(img_3d_channels_first)
    z_slices = np.arange(start=0, stop=img_3d_channels_first.shape[0], step=1)
    chosen_slices = z_slices[chosen_z_layer - (num_z_slices / 2):chosen_z_layer + (num_z_slices / 2)]
    slices_left = num_z_slices - len(chosen_slices)
    if slices_left > 0:
        z_slices = np.delete(z_slices, chosen_slices)
        complementing_slices = np.random.choice(z_slices, size=slices_left, replace=False)
        chosen_slices = np.concatenate((chosen_slices, complementing_slices), axis=0)
    return img_3d_channels_first[chosen_slices]
