import json
from copy import deepcopy

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

import utils

seed = 42
np.random.seed = seed


class DataGenerator(keras.utils.Sequence):
    def __init__(self, patches_path, batch_size, patch_size, data_set='Train'):
        self.brightfield_paths = os.listdir(os.path.join(patches_path, data_set, 'Brightfield'))
        self.fluorescent_paths = os.listdir(os.path.join(patches_path, data_set, 'Fluorescent'))
        self.idx_array = np.arange(start=0, stop=len(self.brightfield_paths), step=1)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.data_set = data_set

    def __len__(self):  # num of batches in epoch
        return len(self.brightfield_paths) // self.batch_size

    def __getitem__(self, idx):
        sampled_indexes = np.random.choice(self.idx_array, self.batch_size, replace=False)
        np.delete(self.idx_array, sampled_indexes)
        brightfield_batch = []
        fluorescent_batch = []
        for img_idx in sampled_indexes:
            curr_brightfield = utils.load_numpy_array(self.brightfield_paths[img_idx])
            curr_fluorescent = utils.load_numpy_array(self.fluorescent_paths[img_idx])
            brightfield_batch.append(curr_brightfield)
            fluorescent_batch.append(curr_fluorescent)
            # np.dstack(fluorescent_batch, curr_fluorescent)

        if self.data_set == 'Train':
            brightfield_batch = self.augment_images(brightfield_batch)
            fluorescent_batch = self.augment_images(fluorescent_batch)

        return np.array(brightfield_batch), np.array(fluorescent_batch)

    def on_epoch_end(self):
        self.idx_array = np.arange(start=0, stop=len(self.brightfield_paths), step=1)

    def augment_images(self, arr):  ##TODO consult Liad on dtype.
        return arr
        # dtype='float8'
        augmentor = ImageDataGenerator(featurewise_center=True, rotation_range=90,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.2, data_format='channels_last')
        # can save augmented image to dir as well
        augmented_batch_gen = augmentor.flow(x=arr, y=None, shuffle=True, seed=seed, batch_size=self.batch_size)
        return augmented_batch_gen.__next__()
