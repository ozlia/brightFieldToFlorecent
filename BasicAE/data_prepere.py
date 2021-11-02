import os
from aicsimageio import AICSImage
import numpy as np
import cv2

photo_limit = 10


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

    # return np.array(bright_field), np.array(fluorescent)
    return bright_field,fluorescent


def load_images_as_batches(brightfield_fluorescent_tiff_paths=None, batch_size=16,img_res=(128,128),sampling=False):
    if brightfield_fluorescent_tiff_paths is None:
        raise ValueError('load batch was not given an array of paths to work with')

    if sampling:
        n_batches = max(int(len(brightfield_fluorescent_tiff_paths) / batch_size),2)  #basically 1 but adjusted for loop
        total_samples = min(n_batches * batch_size,len(brightfield_fluorescent_tiff_paths))
    else:
        n_batches = int(len(brightfield_fluorescent_tiff_paths) / batch_size)
        total_samples = n_batches * batch_size





    # Sample n_batches * batch_size from each path list so that model sees all
    # samples from both domains
    sampled_paths = np.random.choice(brightfield_fluorescent_tiff_paths, total_samples, replace=False)
    total_brightfield, total_fluorescent = [], []
    for i in range(n_batches - 1):
        curr_batch = sampled_paths[i * batch_size : (i + 1) * batch_size]
        curr_brightfield,curr_fluorescent = separate_data(fovs=curr_batch,x=img_res[0],y=img_res[1],z=1)
        total_brightfield += curr_brightfield
        total_fluorescent += curr_fluorescent

        # bright_field_array = np.reshape(curr_brightfield, (len(curr_brightfield), img_res[0], img_res[1], 1))
        # fluorescent_array = np.reshape(curr_fluorescent, (len(curr_fluorescent), img_res[0], img_res[1], 1))
        # yield bright_field_array,fluorescent_array
        # if not curr_brightfield or not curr_fluorescent:
        #     continue
        yield np.array(curr_brightfield),np.array(curr_fluorescent)

    # return np.array(total_brightfield), np.array(total_fluorescent)

        # yield imgs_A, imgs_B
