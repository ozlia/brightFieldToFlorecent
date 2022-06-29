import json
import shutil
from datetime import datetime
from os import path, makedirs, listdir, chdir, getcwd
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from helpers import data_prepare, utils


class DataGeneratorPreparation:  # better as a class - can be easily replaced with regular preparation
    def __init__(self, img_size_channels_first, patch_size_channels_last, org_type, resplit=False,
                 validation_size=0.0, test_size=0.3, initial_testing=False):
        assert type(
            validation_size) == float and 0 <= validation_size < 1, f'Expected float in range [0,1) received: {validation_size}'
        assert type(
            test_size) == float and 0.1 <= test_size <= 0.5, f'Expected float in range [0.1,0.5] received: {test_size}'

        self.org_type = org_type

        self.val_size = validation_size
        self.test_size = test_size

        self.img_size_channels_first = img_size_channels_first
        self.patch_size_channels_last = patch_size_channels_last
        self.patch_size_channels_first = patch_size_channels_last[::-1]  # assuming patch is square

        self.imgs_bulk_size = 25
        self.seed = 42

        self.patches_dir_path = path.join(self.org_type, 'Patches')
        self.images_dir_path = path.join(self.org_type, 'Images')
        self.meta_dir_path = path.join(self.org_type, 'MetaData')
        self.images_mapping_fpath = self.get_mapping_fpath()
        self.patches_meta_data_fpath = self.get_meta_data_fpath()

        self.prepare_images_in_disc_save_only(initial_testing)
        # self.prepare_images_in_disc(resplit)  # flexible incase we don't want to initiate this with DataGeneratorPrep

    def prepare_images_in_disc_save_only(self, initial_testing):
        if path.exists(utils.get_dir(self.images_dir_path)):
            return
        self.prep_dirs()
        self.save_images(initial_testing)

    def prepare_images_in_disc(self, resplit):
        self.prep_dirs()
        user_patches_meta_deta: dict = self.search_users_for_images()
        if user_patches_meta_deta is None:
            self.save_images()
            return

        elif user_patches_meta_deta['User'] != utils.USER:
            self.copy_images(user_patches_meta_deta)  # if patches not copied, make new

        if resplit:
            self.resplit_images(user_patches_meta_deta)

    def copy_images(self, user_meta_data: dict):
        user = user_meta_data['User']

        utils.set_dir(user)
        origin_path = utils.DIRECTORY
        origin_patches_path = utils.get_dir(self.patches_dir_path)
        origin_images_path = utils.get_dir(self.images_dir_path)
        origin_meta_data_path = utils.get_dir(self.meta_dir_path)

        utils.set_dir(utils.USER)
        dest_patches_path = utils.get_dir(self.patches_dir_path)
        dest_images_path = utils.get_dir(self.images_dir_path)
        dest_meta_data_path = utils.get_dir(self.meta_dir_path)

        if user_meta_data['Patch_Size'] == self.patch_size_channels_last:
            print(f'copying images and patches from {user}')
            shutil.copytree(origin_path, utils.DIRECTORY)
        else:  # TODO: Save with new patch_size. Might change to "save patches" to "adapt patch size" later
            print(f'copying images from {user}')
            shutil.copytree(origin_images_path, dest_images_path)  # copy images

            print(f'copying meta data from {user}')
            shutil.copytree(origin_meta_data_path, dest_meta_data_path)  # copy meta data

            print(f'generating patches')
            image_name_to_dataset: pd.DataFrame = pd.read_csv(self.images_mapping_fpath)  # read image mapping
            for row in image_name_to_dataset.iterrows():  # load each image and save its' patches
                for data_set in ['Train', 'Validation', 'Test']:
                    for data_format in ['Brightfield', 'Fluorescent']:
                        img_path = self.build_img_path(base_path=dest_images_path, data_set=data_set,
                                                       data_format=data_format)
                        curr_img = utils.load_numpy_array(path=path.join(img_path, row['Name']))
                        self.save_patches(img=curr_img, img_name=row['Name'], format=data_format, data_set=data_set)

        print(
            f'images copied and patches {"copied" if user_meta_data["matches_patches"] else "generated"} for local user.')

    def train_val_test_split(self, base_names=False, initial_testing=False):  # returns paths for assaf storage
        name_to_dataset_img_paths = {}
        if initial_testing:
            limit = 2
        else:
            limit = None

        imgs_paths = data_prepare.load_paths_v2(self.org_type, limit)

        if base_names:
            imgs_paths = [path.basename(img) for img in imgs_paths]

        # train,val,test split
        X_train, X_test = train_test_split(imgs_paths, test_size=self.test_size,
                                           random_state=self.seed,
                                           shuffle=True)
        name_to_dataset_img_paths['Test'] = X_test
        if self.val_size > 0:
            X_train, X_val = train_test_split(X_train, test_size=self.val_size, random_state=self.seed)
            name_to_dataset_img_paths['Validation'] = X_test

        name_to_dataset_img_paths['Train'] = X_train

        return name_to_dataset_img_paths

    def resplit_images(self, user_meta_data):

        img_name_to_location = pd.read_csv(self.images_mapping_fpath)
        patches_meta_data = pd.read_csv(self.patches_meta_data_fpath)

        name_to_dataset_img_paths = self.train_val_test_split(base_names=True)
        name_to_dataset_img_paths = {dataset_name: set(l_img_paths) for dataset_name, l_img_paths in
                                     name_to_dataset_img_paths.items()}
        new_data_set_mapping = []

        for row in img_name_to_location.iterrows():
            # find new dataset based on split
            for dataset_name in name_to_dataset_img_paths.keys():
                if row[1]['Name'] in name_to_dataset_img_paths[dataset_name]:
                    new_data_set_mapping.append(dataset_name)
                    break

            assert len(img_name_to_location) == len(
                new_data_set_mapping), f"An image from a dataset doesn't appear in both mappings"

            for data_format in ['Brightfield', 'Fluorescent']:
                # move images
                prev_img_path = self.build_img_path(base_path=self.images_dir_path, data_set=new_data_set_mapping[i],
                                                    format=data_format)
                new_img_path = self.build_img_path(base_path=self.images_dir_path, data_set=row['Data_Set'],
                                                   format=data_format)
                shutil.move(src=path.join(prev_img_path, row['Name']), dst=path.join(new_img_path, row['Name']))

                # move patches
                img_name = row['Name'].split('.')[0]
                for i in range(patches_meta_data['Number_Of_Patches']):
                    shutil.move(src=path.join(prev_img_path, f"{img_name}_{i}.png"),
                                dst=path.join(new_img_path, f"{img_name}_{i}.png"))

            # save the new data into the files
            img_name_to_location['Data_Set'] = new_data_set_mapping
            img_name_to_location.to_csv(self.images_mapping_fpath)
            self.save_patches_meta_data()

    def save_images(self, initial_testing):
        imgs_names = []
        imgs_data_set = []
        name_to_dataset_img_paths = self.train_val_test_split(base_names=False, initial_testing=initial_testing)
        print(
            f'Proccesing a tiff should take upto 25 seconds, so loading each image batch should take upto {25 * self.imgs_bulk_size // 60} minutes')
        for data_set_name, data_set_paths in name_to_dataset_img_paths.items():
            # if too many imgs to read at once
            for i in range(0, len(data_set_paths), self.imgs_bulk_size):
                curr_data_set_paths = data_set_paths[i:i + self.imgs_bulk_size]
                start = datetime.now()
                curr_batch_num = i // self.imgs_bulk_size + 1
                print(
                    f'Starting to process batch {curr_batch_num} in {data_set_name} set. Current time: {start}')
                data_sets = data_prepare.separate_data(curr_data_set_paths, self.patch_size_channels_first)
                print(
                    f'Loading batch number {curr_batch_num} took: {utils.get_time_diff_minutes(datetime.now(), start)} minutes')
                brightfield_arr, fluorescent_arr = (utils.transform_dimensions(data_set, [0, 2, 3, 1]) for data_set in
                                                    data_sets)  # costly operation

                start = datetime.now()
                print(f'Saving images and patches for batch {curr_batch_num}. Current time: {start}')
                for i, (bf_img, flr_img) in enumerate(zip(brightfield_arr, fluorescent_arr)):
                    curr_img_name = path.basename(curr_data_set_paths[i]).split('.')[0]
                    imgs_names.append(curr_img_name)
                    img_dir = self.build_img_path(self.images_dir_path, data_set_name, 'Brightfield')
                    if path.exists(utils.get_dir(path.join(img_dir, f'{curr_img_name}.npy'))):
                        continue

                    self.save_img_and_patches(img=bf_img, img_name=curr_img_name, format='Brightfield',
                                              data_set=data_set_name)
                    self.save_img_and_patches(img=flr_img, img_name=curr_img_name, format='Fluorescent',
                                              data_set=data_set_name)
                print(
                    f'Saving images and patches for batch number {curr_batch_num} took {utils.get_time_diff_minutes(datetime.now(), start)} minutes')

                imgs_data_set += [data_set_name] * brightfield_arr.shape[0]
        self.save_patches_meta_data()
        img_name_to_dataset = pd.DataFrame(data=dict(Name=imgs_names, Data_Set=imgs_data_set))
        img_name_to_dataset.to_csv(path_or_buf=self.images_mapping_fpath)

    def save_img_and_patches(self, img, img_name, format, data_set):
        img_path = self.build_img_path(self.images_dir_path, data_set, format)
        utils.save_numpy_array(array=img, path=path.join(img_path, img_name))
        self.save_patches(img=img, img_name=img_name, format=format, data_set=data_set)

    def save_patches(self, img, img_name, format, data_set):
        img_name = img_name.split('.')[0]
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)  # for patchify process
        patches = utils.utils_patchify(img, self.patch_size_channels_last, resize=True, over_lap_steps=1)
        patch_path = self.build_img_path(self.patches_dir_path, data_set, format)
        for i, patch in enumerate(patches):
            utils.save_numpy_array(array=patch, path=path.join(patch_path, f'{img_name}_{i}'))

    def save_patches_meta_data(self):
        num_patches_in_img = (self.img_size_channels_first[1] // self.patch_size_channels_last[0]) * (
                self.img_size_channels_first[2] // self.patch_size_channels_last[1])
        total_num_patches = num_patches_in_img * len(listdir(utils.get_dir(self.images_dir_path)))
        patches_meta_deta = dict(Number_Of_Patches=total_num_patches, Patches_In_Image=num_patches_in_img,
                                 Patch_Size=self.patch_size_channels_last)

        with open(self.patches_meta_data_fpath, 'w') as meta_data_file:
            json.dump(patches_meta_deta, meta_data_file)

    def build_img_path(self, base_path, data_set, format):
        return path.join(base_path, data_set, format)

    def get_meta_data_fpath(self):
        return utils.get_dir(path.join(self.meta_dir_path, 'PatchesMetaData.json'))

    def get_mapping_fpath(self):
        return utils.get_dir(path.join(self.meta_dir_path, 'ImageToLocation.csv'))

    def prep_dirs(self):
        origin_wd = getcwd()
        root_dir = utils.DIRECTORY
        makedirs(root_dir, exist_ok=True)
        chdir(root_dir)

        makedirs(self.meta_dir_path, exist_ok=True)

        for data_set in ['Train', 'Validation', 'Test']:
            for data_format in ['Brightfield', 'Fluorescent']:
                curr_imgs_dir = path.join(self.images_dir_path, data_set, data_format)
                curr_patches_dir = path.join(self.patches_dir_path, data_set, data_format)
                makedirs(curr_imgs_dir, exist_ok=True)
                makedirs(curr_patches_dir, exist_ok=True)

        chdir(origin_wd)

    def search_users_for_images(self):
        '''
        searches all users for images and patches of org_type and patches of patch_size.
        prioritizes curr_user.
        @return: dict: contains meta data (patch_size, num of patches, etc.. ) related to patches saved at user.
        '''
        user_meta_data_patch_no_match = None

        usernames = utils.get_usernames()

        try:
            for user in usernames:
                utils.set_dir(user)
                if not path.exists(self.get_meta_data_fpath()):
                    continue

                with open(self.get_meta_data_fpath()) as f:
                    meta_data_file = json.load(f)

                meta_data_file['User'] = user
                if user_meta_data_patch_no_match is None:
                    user_meta_data_patch_no_match = meta_data_file

                if meta_data_file['Patch_Size'] == self.patch_size_channels_last:
                    return meta_data_file
            return user_meta_data_patch_no_match

        finally:
            utils.set_dir(utils.USER)
