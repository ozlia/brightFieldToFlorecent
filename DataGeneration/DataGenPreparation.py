import json
import shutil
from os import path, makedirs, listdir, chdir, getcwd
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import utils
import data_prepare


class DataGeneratorPreparation:  # better as a class - can be easily replaced with regular preparation
    def __init__(self, patch_size, org_type, batch_size=None, resplit=False,
                 validation_size=0.0, test_size=0.1):
        assert type(validation_size) is float and type(test_size) is float, 'data set sizes must be of type float'
        assert test_size > 0.0 and test_size <= 1, 'You must determine a valid test set size between 0 and 1'

        self.org_type = org_type

        self.val_size = validation_size
        self.test_size = test_size

        self.img_size = (6, 640, 896)
        self.patch_size = patch_size
        self.batch_size = batch_size

        # self.imgs_bulk_size = 150
        self.imgs_bulk_size = 1
        self.seed = 42

        self.patches_dir_path = path.join(self.org_type, 'Patches')
        self.images_dir_path = path.join(self.org_type, 'Images')
        self.meta_dir_path = path.join(self.org_type, 'MetaData')
        self.images_mapping_fpath = self.get_mapping_fpath()
        self.patches_meta_data_fpath = self.get_meta_data_fpath()

        # self.prepare_images_in_disc(resplit)  # flexible incase we don't want to initiate this with DataGeneratorPrep

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

        if user_meta_data['Patch_Size'] == self.patch_size:
            print(f'copying images and patches from {user}')
            shutil.copytree(origin_path, utils.DIRECTORY)
        else:  # Save with new patch_size. Might change to "save patches" to "adapt patch size" later
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
                        # curr_img = utils.load_full_2d_pic(path.join(img_path, row['Name']))
                        curr_img = utils.load_numpy_array(path=path.join(img_path, row['Name']))
                        self.save_patches(img=curr_img, img_name=row['Name'], format=data_format, data_set=data_set)

        print(
            f'images copied and patches {"copied" if user_meta_data["matches_patches"] else "generated"} for local user.')

    def train_val_test_split(self, base_names=False):  # returns paths for assaf storage

        imgs_paths = data_prepare.load_paths_v2(self.org_type)

        if base_names:
            imgs_paths = [path.basename(img) for img in imgs_paths]

        # train,val,test split
        X_train, X_test = train_test_split(imgs_paths, test_size=self.test_size,
                                           random_state=self.seed,
                                           shuffle=True)
        if self.val_size > 0:
            X_train, X_val = train_test_split(X_train, test_size=self.val_size, random_state=self.seed)
        else:
            X_val = []

        return X_train, X_val, X_test

    def resplit_images(self, user_meta_data):

        img_name_to_location = pd.read_csv(self.images_mapping_fpath)
        patches_meta_data = pd.read_csv(self.patches_meta_data_fpath)
        new_data_set_mapping = []

        X_train, X_val, X_test = (set(data_set) for data_set in self.train_val_test_split(base_names=True))

        for i, row in enumerate(img_name_to_location.iterrows()):
            # find new dataset based on split
            if row['Name'] in X_train:
                new_data_set_mapping.append('Train')
            elif row['Name'] in X_val:
                new_data_set_mapping.append('Validation')
            elif row['Name'] in X_test:
                new_data_set_mapping.append('Test')
            else:
                raise ValueError(
                    f"Incorrect mapping - couldn't find image {row['Name']} in any data set")
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

    def save_images(self):  # TODO add os.makedirs
        imgs_names = []
        imgs_data_set = []
        X_train, X_val, X_test = self.train_val_test_split()
        for data_set_paths, data_set_name in zip([X_train, X_val, X_test], ['Train', 'Validation', 'Test']):
            # if too many imgs to read at once
            for i in range(0, len(data_set_paths),self.imgs_bulk_size):
                curr_data_set_paths = data_set_paths[i:i + self.imgs_bulk_size]
                data_sets = data_prepare.separate_data(curr_data_set_paths, self.img_size)
                brightfield_arr, fluorescent_arr = (utils.transform_dimensions(data_set, [0, 2, 3, 1]) for data_set in
                                                    data_sets)

                imgs_data_set += [data_set_name] * len(data_set_paths)

                for i, (bf_img, flr_img) in enumerate(zip(brightfield_arr, fluorescent_arr)):  # TODO test this
                    imgs_names.append(path.basename(data_set_paths[i]))
                    self.save_img_and_patches(img=bf_img, name=imgs_names[i], format='Brightfield',
                                              data_set=data_set_name)
                    self.save_img_and_patches(img=flr_img, name=imgs_names[i], format='Fluorescent',
                                              data_set=data_set_name)

            pd.DataFrame(data=dict(Name=imgs_names, Data_Set=imgs_data_set)).to_csv(self.images_mapping_fpath)

    def save_img_and_patches(self, img, img_name, format, data_set):
        img_path = self.build_img_path(self.images_dir_path, data_set, format)
        utils.save_numpy_array(array=img,path=path.join(img_path, img_name))
        # utils.save_full_2d_pic(img=img, name=path.join(img_path, img_name))
        self.save_patches(img=img, img_name=img_name, format=format, data_set=data_set)

    def save_patches(self, img, img_name: str, format, data_set):
        img_name = img_name.split('.')[0]
        patches = utils.utils_patchify(img, self.patch_size, resize=True, over_lap_steps=1)
        patch_path = self.build_img_path(self.patches_dir_path, data_set, format)
        for row in patches:
            for i, patch in enumerate(row):
                utils.save_numpy_array(array=img, path=path.join(patch_path, f'{img_name}_{i}.png'))
                # utils.save_full_2d_pic(img=patch, name=path.join(patch_path, f'{img_name}_{i}.png'))

        self.save_patches_meta_data()

    def save_patches_meta_data(self):
        num_patches_in_img = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        total_num_patches = num_patches_in_img * len(listdir(self.images_dir_path))
        patches_meta_deta = dict(Number_Of_Patches=total_num_patches, Patches_In_Image=num_patches_in_img,
                                 Patch_Size=self.patch_size)

        with open(self.patches_meta_data_fpath, 'w') as meta_data_file:
            json.dump(patches_meta_deta, meta_data_file)

    def build_img_path(self, base_path, data_set, format):
        return path.join(base_path, data_set, format)

    def get_meta_data_fpath(self):
        return utils.get_dir(path.join(self.patches_dir_path, 'PatchesMetaData.json'))

    def get_mapping_fpath(self):
        return utils.get_dir(path.join(self.images_dir_path, 'ImageToLocation.csv'))

    def prep_dirs(self):
        origin_wd = getcwd()
        root_dir = utils.DIRECTORY
        makedirs(root_dir,exist_ok=True)
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

                if meta_data_file['Patch_Size'] == self.patch_size:
                    return meta_data_file
            return user_meta_data_patch_no_match

        finally:
            utils.set_dir(utils.USER)
