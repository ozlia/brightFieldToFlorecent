from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from DataGeneration.DataGenerator.DataGen import DataGenerator


class TestDataGenerator(DataGenerator):

    def __len__(self):
        return len(self.brightfield_imgs_paths)

    def __getitem__(self, batch_i):
        if self.send_brightfield_img:
            brightfield_patches_batch = np.load(self.brightfield_imgs_paths[batch_i])
        else:
            brightfield_img_patches_paths = self.brightfield_patches_paths[
                                            batch_i * self.num_patches_in_img: (batch_i + 1) * self.num_patches_in_img]
            brightfield_patches_batch = np.array(
                [np.load(patch_path) for patch_path in brightfield_img_patches_paths])

        fluorescent_img = np.load(self.fluorescent_imgs_paths[batch_i])
        return brightfield_patches_batch, fluorescent_img

    def on_epoch_end(self):
        pass

    def augment_images(self, arr):
        # dtype='float16 or float64 depending on prep'
        augmentor = ImageDataGenerator(featurewise_center=True, rotation_range=90,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.2, data_format='channels_last')
        # can save augmented image to dir as well
        augmented_batch_gen = augmentor.flow(x=arr, y=None, shuffle=True, seed=seed, batch_size=self.batch_size)
        return next(augmented_batch_gen)
