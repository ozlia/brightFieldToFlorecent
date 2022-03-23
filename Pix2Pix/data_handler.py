import gc
import os
import data_prepare
import utils
import numpy as np
from sklearn.model_selection import train_test_split

class data_handler():
    def __init__(self):
        self.brightfield_patches = None
        self.fluorescent_patches = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.img_size = (6, 128, 128)  # (z,y,x)
        self.img_size_rev = (self.img_size[1], self.img_size[2], self.img_size[0])
        self.img_limit = 150

        #prep paths for input images

        # self.org_type = "Nuclear-envelope/"
        self.org_type = "Mitochondria"

        self.input_img_array_path = os.path.join(self.org_type, 'input_images_after_data_prepare_norm')
        self.output_img_array_path = os.path.join(self.org_type, 'output_images_after_data_prepare_norm')

        self.load_prep_images()

    def load_prep_images(self):
        brightfield_imgs, fluorescent_imgs = self.load_images_from_memory()
        train_test_split_tup = train_test_split(brightfield_imgs, fluorescent_imgs,
                                                test_size=0.3,
                                                random_state=3,
                                                shuffle=True)

        self.X_train, self.X_test, self.y_train, self.y_test = [utils.transform_dimensions(arr, [0, 2, 3, 1]) for arr in train_test_split_tup]
        self.X_train = utils.utils_patchify(self.X_train, self.img_size_rev, resize=True, over_lap_steps=1)
        self.y_train = utils.utils_patchify(self.y_train, self.img_size_rev, resize=True, over_lap_steps=1)

        del brightfield_imgs, fluorescent_imgs, train_test_split_tup
        gc.collect()

    def load_images_from_memory(self):
        arrays_in_memory = len(os.listdir(os.path.join(utils.DIRECTORY,self.org_type))) > 0
        if arrays_in_memory:
            brightfield_imgs = utils.load_numpy_array(self.input_img_array_path + '.npy')
            fluorescent_imgs = utils.load_numpy_array(self.output_img_array_path + '.npy')
        else:
            images_paths = data_prepare.load_paths(self.org_type + '/', limit=self.img_limit)
            brightfield_imgs, fluorescent_imgs = data_prepare.separate_data(images_paths, self.img_size)
            utils.save_numpy_array(brightfield_imgs, self.input_img_array_path)
            utils.save_numpy_array(fluorescent_imgs, self.output_img_array_path)

        return brightfield_imgs,fluorescent_imgs


    def load_images_as_batches(self, batch_size=16, sample_size=-1):
        if sample_size > 0:  # meant for saving an output
            batch_size = sample_size
            n_batches = 1
        else:
            n_batches = int(len(self.X_train) / batch_size)  # TODO problem if doesn't divide evenly and no shuffle
        # TODO should I shuffle?
        for i in range(n_batches):  # Sample n_batches * batch_size so that model sees all
            # sampled_indexes = np.random.choice(self.sample_range, batch_size, replace=False)
            sampled_indexes = np.arange(start=i * batch_size, stop=((i + 1) * batch_size))
            brightfield_patches_samples = self.X_train[sampled_indexes]
            fluorescent_patches_samples = self.y_train[sampled_indexes]
            yield brightfield_patches_samples, fluorescent_patches_samples

