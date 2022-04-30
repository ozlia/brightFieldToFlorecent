import json

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

import utils


class ExistingConfigurationDataGenerator(keras.utils.Sequence):
    def __init__(self, b_path, f_path, batch_size, patch_size, num_patches):
        self.brightfield_array_path = b_path
        self.fluorescent_array_path = f_path
        self.num_patches = num_patches
        self.batch_size = batch_size
        self.patch_size = patch_size

    def __len__(self):  # num of batches in epoch
        return self.num_patches // self.batch_size

    def __getitem__(self, idx):
        brightfield_batch, fluorescent_batch = self.data_gen.__next__()
        return brightfield_batch, fluorescent_batch

    # only works with images
    def build_data_gen(self, data_set):
        '''

        @param brightfield_path: dir path of separate grayscale (1 channel) brightfield PATCHES.
        @param fluorescent_path: dir path of separate grayscale (1 channel) fluorescent PATCHES.
        @param batch_size:
        @param patch_size:
        @return: data generator to insert into fit.
        '''
        if data_set == 'Train':
            aug_args = dict(featurewise_center=True,
                            rotation_range=90,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2)
            shuffle = True
        else:
            aug_args = dict()
            shuffle = False

        patches_dir = utils.get_dir(os.path.join(self.org_type, 'Patches'))
        brightfield_path = os.path.join(patches_dir, data_set, 'Brightfield')
        fluorescent_path = os.path.join(patches_dir, data_set, 'Fluorescent')

        augmentor = ImageDataGenerator(data_format='channels_last', **aug_args)

        data_gen_args = dict(class_mode=None, color_mode='grayscale', seed=self.seed, batch_size=self.batch_size,
                             shuffle=shuffle,
                             target_size=self.patch_size)

        brightfield_data_gen = augmentor.flow_from_directory(directory=brightfield_path, **data_gen_args)
        fluorescent_data_gen = augmentor.flow_from_directory(directory=fluorescent_path, **data_gen_args)

        combined_data_gen = zip(brightfield_data_gen, fluorescent_data_gen)
        return combined_data_gen
