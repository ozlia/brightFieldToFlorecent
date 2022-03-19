import gc
import os

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
        self.img_limit = 150

        # self.org_type = "Nuclear-envelope/"
        self.org_type = "Mitochondria/"
        self.input_img_array_path = os.path.join(self.org_type, 'input_images_after_data_prepare_norm')
        self.output_img_array_path = os.path.join(self.org_type, 'output_images_after_data_prepare_norm')

        self.load_prep_images()

    def get_patches(self, data_input=None, data_output=None):
        data_input = utils.transform_dimensions(data_input, [0, 2, 3, 1])
        data_output = utils.transform_dimensions(data_output, [0, 2, 3, 1])
        self.patches_input = utils.utils_patchify(data_input, self.img_size_rev, resize=True, over_lap_steps=1)
        self.patches_output = utils.utils_patchify(data_output, self.img_size_rev, resize=True, over_lap_steps=1)

    def load_prep_images(self):
        data_input,data_output = self.load_images_from_memory()
        self.get_patches(data_input, data_output)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.patches_input, self.patches_output,
                                                                                test_size=0.3,
                                                                                random_state=3,
                                                                                shuffle=True)
        del self.patches_input, self.patches_output,data_input,data_output #,self.test_x, self.test_y
        gc.collect()

    def load_images_from_memory(self):
        if len(os.listdir(DIRECTORY)) == 0:
            images_paths = data_prepare.load_paths(self.org_type, limit=self.img_limit)
            data_input, data_output = data_prepare.separate_data(images_paths, self.img_size)
            utils.save_numpy_array(data_input, self.input_img_array_path)
            utils.save_numpy_array(data_output, self.output_img_array_path)
        else:
            data_input = utils.load_numpy_array(self.input_img_array_path + '.npy')
            data_output = utils.load_numpy_array(self.output_img_array_path + '.npy')

        return data_input,data_output


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