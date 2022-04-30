from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from glob import glob

seed = 42
np.random.seed = seed


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_root_path, batch_size, num_patches_in_img, data_set='Train', data_format_in_disc='npy'):
        # self.fluorescent_paths = os.listdir(os.path.join(patches_path, data_set, 'Fluorescent'))
        patches_root_dir = os.path.join(data_root_path, 'Patches', data_set)
        imgs_root_dir = os.path.join(data_root_path, 'Images', data_set)
        self.brightfield_patches_paths = glob(os.path.join(patches_root_dir, 'Brightfield', f'*.{data_format_in_disc}'))
        self.fluorescent_patches_paths = glob(os.path.join(patches_root_dir, 'Fluorescent', f'*.{data_format_in_disc}'))
        self.fluorescent_imgs_paths = glob(os.path.join(imgs_root_dir, 'Fluorescent', f'*.{data_format_in_disc}'))

        self.idx_array = np.arange(start=0, stop=len(self.brightfield_patches_paths), step=1)
        self.batch_size = batch_size
        self.num_patches_in_img = num_patches_in_img
        self.data_set = data_set

    def __len__(self):
        num_patches_all_imgs = len(self.brightfield_patches_paths)
        if self.data_set == 'Test':  # num imgs
            return num_patches_all_imgs // self.num_patches_in_img
        else:  # num_batches in epoch
            return num_patches_all_imgs // self.batch_size

    def __getitem__(self, idx):
        # assert idx < len(self), f"Expected {len(self)} batches in epoch, received {idx}"
        brightfield_patches_batch = []

        if self.data_set == 'Test':
            sampled_indexes = self.idx_array[
                              idx * self.num_patches_in_img : (idx + 1) * self.num_patches_in_img]
            for img_idx in sampled_indexes:
                curr_brightfield = np.load(self.brightfield_patches_paths[img_idx])
                brightfield_patches_batch.append(curr_brightfield)

            fluorescent_img = np.load(self.fluorescent_imgs_paths[idx])
            return np.array(brightfield_patches_batch), fluorescent_img

        else:  # get random patches from any number of images
            fluorescent_patches_batch = []
            sampled_indexes = np.random.choice(self.idx_array, self.batch_size, replace=False)
            np.delete(self.idx_array, sampled_indexes)

            for img_idx in sampled_indexes:
                curr_brightfield = np.load(self.brightfield_patches_paths[img_idx])
                brightfield_patches_batch.append(curr_brightfield)

                curr_fluorescent = np.load(self.fluorescent_patches_paths[img_idx])
                fluorescent_patches_batch.append(curr_fluorescent)

            # brightfield_patches_batch = self.augment_images(brightfield_patches_batch)
            # fluorescent_patches_batch = self.augment_images(fluorescent_patches_batch)

            return np.array(brightfield_patches_batch), np.array(fluorescent_patches_batch)

    def on_epoch_end(self):
        self.idx_array = np.arange(start=0, stop=len(self.brightfield_patches_paths), step=1)

    def augment_images(self, arr):
        return arr
        # dtype='float16 or float64 depending on prep'
        augmentor = ImageDataGenerator(featurewise_center=True, rotation_range=90,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.2, data_format='channels_last')
        # can save augmented image to dir as well
        augmented_batch_gen = augmentor.flow(x=arr, y=None, shuffle=True, seed=seed, batch_size=self.batch_size)
        return next(augmented_batch_gen)
