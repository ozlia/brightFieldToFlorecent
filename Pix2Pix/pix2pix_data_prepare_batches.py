import math
import os
import sys

import data_prepere
import utils
import numpy as np
import getpass
from sklearn.model_selection import train_test_split

USER = getpass.getuser().split("@")[0]
DIRECTORY = "/home/%s/prediction3D" % USER
os.makedirs(DIRECTORY, exist_ok=True)


class pix2pix_data_prepare():
    def __init__(self, batch_size=128):
        self.batch_size = 16

        self.patches_batches_dirs_paths = dict()

        self.img_size = (6, 128, 128)  # (z,y,x)
        self.img_size_rev = (self.img_size[1], self.img_size[2], self.img_size[0])

        self.limit = 2 #150
        self.org_type = "Mitochondria/"
        # self.num_images_loaded_to_memory = 30
        self.num_images_loaded_to_memory = 1

        self.setup_batches_patches()

        self.brightfield_train_paths = os.listdir(os.path.join(DIRECTORY,self.patches_batches_dirs_paths[f'brightfield_train']))
        self.fluorescent_train_paths = os.listdir(os.path.join(DIRECTORY,self.patches_batches_dirs_paths[f'fluorescent_train']))

    def load_images_as_batches(self, batch_size=16,sample_size=-1):

        if sample_size != -1:
            brightfield_sample = utils.load_numpy_array(self.brightfield_train_paths[0])[:sample_size]
            fluorescent_sample = utils.load_numpy_array(self.fluorescent_train_paths[0])[:sample_size]
            yield brightfield_sample, fluorescent_sample
        else:

            for brightfield_batch_path, fluorescent_batch_path in zip(self.brightfield_train_paths,
                                                                      self.fluorescent_train_paths):
                brightfield_batch = utils.load_numpy_array(os.path.join(self.patches_batches_dirs_paths['brightfield_train'],brightfield_batch_path))
                fluorescent_batch = utils.load_numpy_array(os.path.join(self.patches_batches_dirs_paths['fluorescent_train'],fluorescent_batch_path))
                yield brightfield_batch, fluorescent_batch

    def save_batches_to_disc(self):
        imgs_paths_of_org_type = data_prepere.load_paths(self.org_type, limit=self.limit)
        train_paths, test_paths = train_test_split(imgs_paths_of_org_type, test_size=0.2, random_state=3)
        self.save_images_of_specific_organelle(imgs_paths=train_paths, train_test_dir_name='train')
        self.save_images_of_specific_organelle(imgs_paths=test_paths, train_test_dir_name='test')

    def calc_num_batches(self, origin_img_dims,
                         num_imgs_loaded):  # TODO not entirely accurate because of patching method
        num_patches_in_single_img = (origin_img_dims[0] * origin_img_dims[1]) / (
                self.img_size_rev[0] * self.img_size_rev[1])
        total_num_patches = num_patches_in_single_img * num_imgs_loaded
        return int(total_num_patches / self.batch_size)

    def save_images_of_specific_organelle(self, imgs_paths=None, train_test_dir_name=''):
        if imgs_paths is None or not train_test_dir_name or f'brightfield_{train_test_dir_name}' not in self.patches_batches_dirs_paths:
            return

        num_iters = int(math.ceil((len(imgs_paths) / self.num_images_loaded_to_memory)))
        num_batches_patches = self.calc_num_batches(origin_img_dims=(896,640,6),
                                                    num_imgs_loaded=self.num_images_loaded_to_memory)

        brightfield_target_dir_name = self.patches_batches_dirs_paths[f'brightfield_{train_test_dir_name}']
        fluorescent_target_dir_name = self.patches_batches_dirs_paths[f'fluorescent_{train_test_dir_name}']

        for i in range(num_iters):
            start_idx_imgs = i * num_iters
            brightfield_img_batch, fluorescent_img_batch = data_prepere.separate_data(
                imgs_paths[start_idx_imgs:start_idx_imgs + self.num_images_loaded_to_memory],
                self.img_size)
            brightfield_img_batch = utils.transform_dimensions(brightfield_img_batch, [0, 2, 3, 1])
            fluorescent_img_batch = utils.transform_dimensions(fluorescent_img_batch, [0, 2, 3, 1])

            brightfield_patches = utils.utils_patchify(brightfield_img_batch, self.img_size_rev, resize=True,
                                                       over_lap_steps=1)
            fluorescent_patches = utils.utils_patchify(fluorescent_img_batch, self.img_size_rev, resize=True,
                                                       over_lap_steps=1)

            for j in range(num_batches_patches):
                brightfield_patch_batch_path = os.path.join(brightfield_target_dir_name, str(j))
                fluorescent_patch_batch_path = os.path.join(fluorescent_target_dir_name, str(j))
                start_idx_patches = j * num_batches_patches

                utils.save_numpy_array(brightfield_patches[start_idx_patches:start_idx_patches + self.batch_size],
                                       brightfield_patch_batch_path)
                utils.save_numpy_array(fluorescent_patches[start_idx_patches:start_idx_patches + self.batch_size],
                                       fluorescent_patch_batch_path)

    def setup_batches_patches(self):
        dirs_exist = False
        for b_f in ['brightfield', 'fluorescent']:
            for test_train in ['test', 'train']:
                try:
                    target_path = os.path.join(self.org_type[:-1], test_train, str(self.batch_size), b_f)
                    self.patches_batches_dirs_paths[f'{b_f}_{test_train}'] = target_path
                    os.makedirs(os.path.join(DIRECTORY, target_path))
                except OSError:
                    dirs_exist = True
                    continue

        if not dirs_exist:
            self.save_batches_to_disc()
