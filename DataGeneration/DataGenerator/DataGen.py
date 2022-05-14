from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from glob import glob
import pandas as pd

seed = 3
np.random.seed = seed


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_root_path, meta_data_fpath, num_epochs, batch_size, num_patches_in_img,
                 data_set_type='Train',
                 data_format_in_disc='npy'):
        self.num_patches_in_img = num_patches_in_img
        self.set_paths(data_root_path, meta_data_fpath, data_set_type, data_format_in_disc)

        self.idx_array = np.arange(start=0, stop=len(self.brightfield_patches_paths), step=1)
        self.batch_size = batch_size
        self.data_set_type = data_set_type

        # Change if you want to return a brightfield img on test set or brightfield patches
        self.send_brightfield_img = False
        self.num_epochs_passed = 0
        idx_matrix_shape = (num_epochs, self.num_batches_in_epoch, batch_size)
        self.idx_matrix = np.random.randint(low=0, high=len(self.brightfield_patches_paths),
                                            size=idx_matrix_shape).reshape(idx_matrix_shape)

    def set_paths(self, data_root_path, meta_data_fpath, data_set_type, data_format_in_disc):
        img_name_to_dataset = pd.read_csv(meta_data_fpath)
        img_name_to_dataset = img_name_to_dataset.loc[img_name_to_dataset['Data_Set'] == data_set_type]

        fix_naming_imgs = lambda root_path, data_format: [
            os.path.join(root_path, data_format, f'{name}.{data_format_in_disc}') for name in
            img_name_to_dataset['Name']]
        fix_naming_patches = lambda root_path, data_format: [
            os.path.join(root_path, data_format, f'{name}_{i}.{data_format_in_disc}') for name in
            img_name_to_dataset['Name'] for i in range(self.num_patches_in_img)]

        patches_root_dir = os.path.join(data_root_path, 'Patches', data_set_type)
        imgs_root_dir = os.path.join(data_root_path, 'Images', data_set_type)

        self.brightfield_patches_paths = fix_naming_patches(patches_root_dir, 'Brightfield')
        self.fluorescent_patches_paths = fix_naming_patches(patches_root_dir, 'Fluorescent')
        self.brightfield_imgs_paths = fix_naming_imgs(imgs_root_dir, 'Brightfield')
        self.fluorescent_imgs_paths = fix_naming_imgs(imgs_root_dir, 'Fluorescent')

    @property
    def num_batches_in_epoch(self):
        return len(self.brightfield_patches_paths) // self.batch_size + 1

    def __len__(self):
        return len(self.brightfield_imgs_paths) if self.data_set_type == 'Test' else self.num_batches_in_epoch

    def __getitem__(self, batch_i):
        # assert idx < len(self), f"Expected {len(self)} batches in epoch, received {idx}"

        if self.data_set_type == 'Test':  # get consecutive brightfield patches and one matching floro img
            if self.send_brightfield_img:
                brightfield_patches_batch = np.load(self.brightfield_imgs_paths[batch_i])
            else:
                brightfield_img_patches_paths = self.brightfield_patches_paths[
                                                batch_i * self.num_patches_in_img: (
                                                                                               batch_i + 1) * self.num_patches_in_img]
                brightfield_patches_batch = np.array(
                    [np.load(patch_path) for patch_path in brightfield_img_patches_paths])

            fluorescent_img = np.load(self.fluorescent_imgs_paths[batch_i])
            return brightfield_patches_batch, fluorescent_img

        else:  # get random patches from any number of images
            fluorescent_patches_batch = []
            brightfield_patches_batch = []
            sampled_indexes = self.idx_matrix[self.num_epochs_passed, batch_i, :]

            for img_idx in sampled_indexes:
                curr_brightfield = np.load(self.brightfield_patches_paths[img_idx])
                brightfield_patches_batch.append(curr_brightfield)

                curr_fluorescent = np.load(self.fluorescent_patches_paths[img_idx])
                fluorescent_patches_batch.append(curr_fluorescent)

            # brightfield_patches_batch = self.augment_images(brightfield_patches_batch)
            # fluorescent_patches_batch = self.augment_images(fluorescent_patches_batch)

            return np.array(brightfield_patches_batch), np.array(fluorescent_patches_batch)

    def on_epoch_end(self):
        self.num_epochs_passed += 1

    def augment_images(self, arr):
        # dtype='float16 or float64 depending on prep'
        augmentor = ImageDataGenerator(featurewise_center=True, rotation_range=90,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.2, data_format='channels_last')
        # can save augmented image to dir as well
        augmented_batch_gen = augmentor.flow(x=arr, y=None, shuffle=True, seed=seed, batch_size=self.batch_size)
        return next(augmented_batch_gen)
