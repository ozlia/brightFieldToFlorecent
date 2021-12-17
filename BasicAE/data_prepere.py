import os
from aicsimageio import AICSImage
import numpy as np
import cv2

photo_limit = 12


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



def separate_data(fovs, x, y,z=1):
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
        img_2D_color = np.expand_dims(img_2D_norm, axis=-1) ## (624, 928) -> ((624, 928, 1))
        img_resized = cv2.resize(img_2D_color, (x, y))  ## (128, 128, 1)
        bright_field.append(img_resized)

        img_2D = img[n_channels - 2, mid_slice, :, :]
        img_2D_norm = (img_2D - np.min(img_2D)) / (np.max(img_2D) - np.min(img_2D))
        img_2D_color = np.expand_dims(img_2D_norm, axis=-1)  ## (624, 928) -> ((624, 928, 1))
        img_resized = cv2.resize(img_2D_color, (x, y))  ## (128, 128, 1)
        fluorescent.append(img_resized)

    # bright_field_array = np.reshape(bright_field, (len(bright_field), x, y, z))
    # fluorescent_array = np.reshape(fluorescent, (len(fluorescent), x, y, z))

    return np.array(bright_field), np.array(fluorescent)
    # return bright_field,fluorescent


def load_images_as_batches(brightfield_fluorescent_tiff_paths=None, batch_size=16,img_res=(128,128),sample_size=-1):

    if not brightfield_fluorescent_tiff_paths:
        raise ValueError('load batch was not given an array of paths to work with')
    # Sample n_batches * batch_size from each path list so that model sees all
    elif sample_size > 0:
        batch_size = sample_size
        n_batches = 1
    else:
        n_batches = max(int(len(brightfield_fluorescent_tiff_paths) / batch_size),1)
    for i in range(n_batches):
        sampled_paths = np.random.choice(brightfield_fluorescent_tiff_paths, batch_size, replace=False)
        # curr_batch = sampled_paths[i * batch_size: (i + 1) * batch_size]
        yield separate_data(fovs=sampled_paths, x=img_res[0], y=img_res[1], z=1)

