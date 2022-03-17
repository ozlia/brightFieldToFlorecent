import os
import sys

import data_prepare
import utils
import numpy as np
import getpass
from sklearn.model_selection import train_test_split

USER = getpass.getuser().split("@")[0]
DIRECTORY = "/home/%s/prediction3D" % USER
os.makedirs(DIRECTORY, exist_ok=True)


class pix2pix_data_prepare():
    def __init__(self):
        self.patches_input = None
        self.patches_output = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

        self.img_size = (6, 128, 128)  # (z,y,x)
        self.img_size_rev = (self.img_size[1], self.img_size[2], self.img_size[0])
        self.saved_input_imgs_fname = 'input_images_after_data_prepare_norm.npy'
        self.saved_output_imgs_fname = 'output_images_after_data_prepare_norm.npy'

        self.limit = 100
        # self.org_type = "Mitochondria/"
        self.org_type = "Nuclear-envelope/"
        self.images_paths = data_prepare.load_paths(self.org_type, limit=self.limit)

        # self.sample_range = np.arange(len(self.train_x))

        self.load_images()


    def load_all_images_of_specific_organelle(self, data_input=None, data_output=None):
        if data_input is None or data_output is None:
            data_input = utils.load_numpy_array(self.saved_input_imgs_fname)
            data_output = utils.load_numpy_array(self.saved_output_imgs_fname)

        data_input = utils.transform_dimensions(data_input, [0, 2, 3, 1])
        data_output = utils.transform_dimensions(data_output, [0, 2, 3, 1])

        self.patches_input = utils.utils_patchify(data_input, self.img_size_rev, resize=True, over_lap_steps=1)
        self.patches_output = utils.utils_patchify(data_output, self.img_size_rev, resize=True, over_lap_steps=1)

    def load_images_as_batches(self, batch_size=16, sample_size=-1):
        if sample_size > 0:  # meant for saving an output
            batch_size = sample_size
            n_batches = 1
        else:
            n_batches = int(len(self.train_x) / batch_size)  # TODO problem if doesn't divide evenly and no shuffle
        # TODO should I shuffle?
        for i in range(n_batches):  # Sample n_batches * batch_size so that model sees all
            # sampled_indexes = np.random.choice(self.sample_range, batch_size, replace=False)
            sampled_indexes = np.arange(start=i * batch_size, stop=((i + 1) * batch_size))
            brightfield_patches_samples = self.train_x[sampled_indexes]
            fluorescent_patches_samples = self.train_y[sampled_indexes]
            yield brightfield_patches_samples, fluorescent_patches_samples

    def load_images(self):
        data_input = None
        data_output = None
        if len(os.listdir(DIRECTORY)) == 0:
            data_input, data_output = self.save_images_of_specific_organelle()
        self.load_all_images_of_specific_organelle(data_input, data_output)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.patches_input, self.patches_output,
                                                                                test_size=0.3,
                                                                                random_state=3,
                                                                                shuffle=True)
        self.patches_input = None
        self.patches_output = None
        self.test_x = None
        self.test_y = None

    def save_images_of_specific_organelle(self):
        data_input, data_output = data_prepare.separate_data(self.images_paths,self.img_size)
        utils.save_numpy_array(data_input, "input_images_after_data_prepare")
        utils.save_numpy_array(data_output, "output_images_after_data_prepare")
        return data_input, data_output
