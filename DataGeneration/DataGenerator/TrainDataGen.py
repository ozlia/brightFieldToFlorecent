from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from DataGeneration.DataGenerator.DataGen import DataGenerator

class TrainDataGenerator(DataGenerator):
    def __init__(self, **kwargs):
        kwargs['data_set_type'] = 'Train'
        super().__init__(**kwargs)
    def __len__(self):
        return self.num_batches_in_epoch

    def __getitem__(self, batch_i):
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
