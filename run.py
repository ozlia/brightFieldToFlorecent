import numpy as np

import data_prepare
from sklearn.model_selection import train_test_split
import utils
from CrossDomainAE.crossDomainAE import AutoEncoderCrossDomain
from DataGeneration.DataGenPreparation.BasicDataGenPreparation import BasicDataGeneratorPreparation
from DataGeneration.DataGenerator.TestDataGen import TestDataGenerator
from DataGeneration.DataGenerator.TrainDataGen import TrainDataGenerator
from Img2ImgAE.autoEncoder import AutoEncoder
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import backend as KB
import pandas as pd
from pandas import DataFrame
from argparse import ArgumentParser
# from CrossDomainAE.crossDomainAE import AutoEncoderCrossDomain
from UNET.Unet import Unet


# interpreter_path = /home/<username>/.conda/envs/<env name>/bin/python - change your user !!
# interpreter_path_omer  = /home/omertag/.conda/envs/my_env/bin/python

METADATA_CSV_PATH = "/sise/assafzar-group/assafzar/fovs/metadata.csv"
PATCH_SIZE = (3, 64, 64)  # (x,y,z)
FULL_IMG_SIZE = (3, 640, 896)

def run(dir, model_name, epochs=200, batch_size=32, read_img=False, org_type=None, img_read_limit=150,
        load_model_date=None, over_lap=1, multiply_img_z=1):

    utils.set_dir(dir)
    patch_size_rev = (PATCH_SIZE[1], PATCH_SIZE[2], PATCH_SIZE[0])
    start = datetime.now()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("reading images")

    if read_img and org_type:
        if not org_type[-1] == "/":
            org_type = org_type + "/"
        data_input, data_output = data_prepare.separate_data(data_prepare.load_paths(org_type, limit=img_read_limit),
                                                             PATCH_SIZE, multiply_img_z=multiply_img_z)
        utils.save_numpy_array(data_input, "input_images_after_data_prepare_norm")
        utils.save_numpy_array(data_output, "output_images_after_data_prepare_norm")
        print("Saved successfully numpy array at %s" % utils.DIRECTORY)
    else:
        data_input = utils.load_numpy_array("input_images_after_data_prepare_norm.npy")
        data_output = utils.load_numpy_array("output_images_after_data_prepare_norm.npy")
        print("Loaded successfully numpy array from %s" % utils.DIRECTORY)

    data_input = utils.transform_dimensions(data_input, [0, 2, 3, 1])
    data_output = utils.transform_dimensions(data_output, [0, 2, 3, 1])

    train_x, test_x, train_y, test_y = train_test_split(data_input, data_output, test_size=0.1, random_state=3,
                                                        shuffle=True)
    patches_train_x = utils.utils_patchify(train_x, patch_size_rev, resize=True, over_lap_steps=over_lap)
    patches_train_y = utils.utils_patchify(train_y, patch_size_rev, resize=True, over_lap_steps=over_lap)

    stop = datetime.now()
    print('Done Reading and Patching, Time: ', stop - start)

    # Free up RAM in case the model definition cells were run multiple times
    KB.clear_session()

    print("init model")
    model = create_model(model_name, patch_size_rev=patch_size_rev, epochs=epochs, batch_size=batch_size)

    if not load_model_date:
        print("training model")
        model.train(patches_train_x, patches_train_y, val_set=0.1, model_dir=dir)
        stop = datetime.now()
        print('Done Train, Time: ', stop - start)

    else:
        print("Loading model .....")
        model.load_model(model_dir=load_model_date)
        stop = datetime.now()
        print('Done Load, Time: ', stop - start)

    save_time = datetime.now().strftime("%H-%M_%d-%m-%Y")
    utils.calculate_pearson_for_all_images(model, test_x[:100], test_y[:100],
                                           model_name=model_name, time=save_time, organelle=org_type)

    print("Generate new pic")
    predicted_img = model.predict([test_x[0]])
    predicted_img_smooth = model.predict_smooth([test_x[0]]) # only if you implanted smooth predict
    print("Saving .........")
    utils.save_np_as_tiff(predicted_img, save_time, "predict", model_name)
    utils.save_np_as_tiff(predicted_img_smooth, save_time, "predict_smooth", model_name) # only if you implanted smooth predict
    utils.save_np_as_tiff(test_x[0], save_time, "input", model_name)
    utils.save_np_as_tiff(test_y[0], save_time, "ground_truth", model_name)
    print("... All tiffs saved !!")
    stop = datetime.now()
    print('Done All, Time: ', stop - start)
    utils.reset_dir()

# interpreter_path = /home/omertag/.conda/envs/my_env/bin/python - change your user !!

def print_full(df: DataFrame):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(df)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


def cmd_helper_script():
    load = None
    org = None
    while load not in ["y", "Y", "n", "N"]:
        print("Do you need to load new images? [y/n]")
        load = input()
    if load in ["y", "Y"]:
        print("Please select organelle name for this list:")
        print("---------------")
        matadata_df = pd.read_csv(METADATA_CSV_PATH)
        all_org = list(set(matadata_df['StructureDisplayName']))
        all_org.remove("None")
        for i in range(0, len(all_org), 3):
            print(', '.join(all_org[i:i + 3]))
        print("---------------")
        org = input()
        while org not in all_org:
            print("please enter valid name from the list above:")
            org = input()
    print("how many epochs?")
    epochs = int(input())
    print("what is the batch size?")
    batch_size = int(input())
    read_img = True if load in ["y", "Y"] else False
    print("Directory name?")
    dir_name = input()

    run(dir=dir_name, model_name="img2img", epochs=epochs, batch_size=batch_size, read_img=read_img, org_type=org,
        img_read_limit=150)

    # print_full(matadata_df)
    # print_full((matadata_df.head()))
    # ['StructureDisplayName']
    # ['ChannelNumberBrightfield']


def organelle_list():
    matadata_df = pd.read_csv(METADATA_CSV_PATH)
    all_org = list(set(matadata_df['StructureDisplayName']))
    all_org.remove("None")
    for i in range(0, len(all_org), 3):
        print(', '.join(all_org[i:i + 3]))


def create_model(name: str, patch_size_rev, epochs, batch_size):
    '''
    name - Must be lower case in this function!!!
    '''

    name = name.lower()

    if name == "img2img":
        return AutoEncoder(patch_size_rev, epochs=epochs, batch_size=batch_size)
    elif name == "b2b":
        return AutoEncoderCrossDomain(patch_size_rev, epochs=epochs, batch_size=batch_size)
    elif name == "f2f":
        return AutoEncoderCrossDomain(patch_size_rev, epochs=epochs, batch_size=batch_size)
    elif name == "unet":
        return Unet(patch_size_rev, epochs=epochs, batch_size=batch_size)
    elif name == "pix2pix":
        return None
    else:
        return None
    # case = {
    #     "img2img": AutoEncoder(img_size_rev, epochs=epochs, batch_size=batch_size),
    #     # "crossdomain": AutoEncoderCrossDomain(img_size_rev, epochs=epochs, batch_size=batch_size),
    #     "B2B": AutoEncoderCrossDomain(img_size_rev, epochs=epochs, batch_size=batch_size),
    #     "F2F": AutoEncoderCrossDomain(img_size_rev, epochs=epochs, batch_size=batch_size),
    #     "unet": Unet(img_size_rev, epochs=epochs, batch_size=batch_size),
    #     "pix2pix": None
    # }
    # return case.get(name, None)


def parse_command_line():
    parser = ArgumentParser(description="Getting arguments to run model")
    parser.add_argument("-m", "--model_type", nargs=1, required=True, help="Model to run [img2img, crossdomain, unet]")
    parser.add_argument("-d", "--dir", default=["run_%s" % datetime.now().strftime("%d-%m-%Y_%H-%M")],
                        nargs=1, help="Directory name to save")
    parser.add_argument("-e", "--epochs", nargs=1, default=1000, type=int, help="Number of epochs")
    parser.add_argument("-bz", "--batch_size", nargs=1, default=32, type=int, help='Batch size')
    parser.add_argument("-ri", "--read_img", action='store_true', default=False,
                        help='Use this flag to read new images')
    parser.add_argument("-o", "--org_type", nargs=1, default=None, type=str,
                        help='Name of Organelle from list: %s' % ', '.join(
                            sorted(set(pd.read_csv(METADATA_CSV_PATH)['StructureDisplayName']))))
    parser.add_argument("-rl", "--read_limit", default=150, type=int, help='Maximum number of images to read')
    args = parser.parse_args()
    run(dir=args.dir, model_name=args.model_type, epochs=args.epochs, batch_size=args.batch_size,
        read_img=args.read_img, org_type=args.org_type[0], img_read_limit=args.read_limit)


def run_with_data_gen(dir, model_name, epochs=200, batch_size=32, read_img=False, org_type=None, img_read_limit=150,
        load_model_date=None, over_lap=1, multiply_img_z=1):
    utils.set_dir(dir)
    img_size_rev = (PATCH_SIZE[1], PATCH_SIZE[2], PATCH_SIZE[0])
    start = datetime.now()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("reading images")

    validation_size = 0.0
    test_size = 0.15
    first_time_testing_if_works = False
    num_patches_in_img = (FULL_IMG_SIZE[1] // img_size_rev[0]) * (FULL_IMG_SIZE[2] // img_size_rev[1])

    dgp = BasicDataGeneratorPreparation(img_size_channels_first=FULL_IMG_SIZE,
                                        patch_size_channels_last=img_size_rev, org_type=org_type,
                                        resplit=False, validation_size=validation_size,
                                        test_size=test_size, initial_testing=first_time_testing_if_works)
    print('done saving, starting to allocate data gens')
    train_data_gen = TrainDataGenerator(meta_data_fpath=dgp.images_mapping_fpath,
                                        data_root_path=utils.get_dir(org_type), num_epochs=epochs,
                                        batch_size=batch_size, num_patches_in_img=num_patches_in_img)
    test_data_gen = TestDataGenerator(meta_data_fpath=dgp.images_mapping_fpath,
                                      data_root_path=utils.get_dir(org_type), num_epochs=epochs,
                                      batch_size=batch_size, num_patches_in_img=num_patches_in_img)

    stop = datetime.now()
    print('Done Reading and Patching, Time: ', stop - start)

    # Free up RAM in case the model definition cells were run multiple times
    KB.clear_session()

    print("init model")
    model = create_model(model_name, patch_size_rev=img_size_rev, epochs=epochs, batch_size=batch_size)

    if not load_model_date:
        print("training model")
        model.train(train_x=train_data_gen, train_label=None, val_set=0.1, model_dir=dir)
        stop = datetime.now()
        print('Done Train, Time: ', stop - start)

    else:
        print("Loading model .....")
        model.load_model(model_dir=load_model_date)
        stop = datetime.now()
        print('Done Load, Time: ', stop - start)

    save_time = datetime.now().strftime("%H-%M_%d-%m-%Y")

    br_path = test_data_gen.brightfield_imgs_paths
    fl_path = test_data_gen.fluorescent_imgs_paths
    test_x = []
    test_y = []
    for br_img, fl_img in zip(br_path, fl_path):
        test_x.append(np.load(br_img))
        test_y.append(np.load(fl_img))
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    utils.calculate_pearson_for_all_images(model, test_x[:100], test_y[:100],
                                           model_name=model_name, time=save_time, organelle=org_type)

    print("Generate new pic")
    predicted_img = model.predict([test_x[0]])
    predicted_img_smooth = model.predict_smooth([test_x[0]])  # only if you implanted smooth predict
    print("Saving .........")
    utils.save_np_as_tiff(predicted_img, save_time, "predict", model_name)
    utils.save_np_as_tiff(predicted_img_smooth, save_time, "predict_smooth",
                          model_name)  # only if you implanted smooth predict
    utils.save_np_as_tiff(test_x[0], save_time, "input", model_name)
    utils.save_np_as_tiff(test_y[0], save_time, "ground_truth", model_name)
    print("... All tiffs saved !!")
    stop = datetime.now()
    print('Done All, Time: ', stop - start)
    utils.reset_dir()



def run_all_orgs(selected_model_name: str, best_orgs_dict: dict):
    """
    Params:
    selected_model - name of the model
    --------------
    best_orgs dict -

    Key: Organelle Name.
    Example - "Mitochondria"

    Value: Model date-time to load, to create new model leave None.
    Example - "28-04-2022_10-59"
    """

    start_all = datetime.now()

    for organelle, model_date in best_orgs_dict.items():

        working_org = "----- Working on organelle: %s -----" % organelle
        print(len(working_org) * "-")
        print(working_org)
        print(len(working_org) * "-")

        model_dir = "/%s_%s_%s/" % (selected_model_name, organelle, model_date) if model_date else None

        run(dir="%s_%s" % (selected_model_name, organelle), model_name=selected_model_name, epochs=50, batch_size=32,
            read_img=True, org_type=organelle, img_read_limit=120, load_model_date=model_dir, multiply_img_z=2)

        done_org = "***** Done organelle: %s *****" % organelle
        print(len(done_org) * "*")
        print(done_org)
        print(len(done_org) * "*")

    stop_all = datetime.now()
    print('All organelles done, Total Time for this run: ', stop_all - start_all)


if __name__ == '__main__':

    # todo please change your run params here
    # see run_all_orgs function and documentation
    # you can copy model name from here
    all_models = ["pix2pix", "unet", "f2f", "b2b", "img2img"]

    selected_model = "unet"
    organelle = "Mitochondria"

    # todo please comment/uncomment your selected Organelle !!
    best_orgs = {
         # "Mitochondria": None,
         "Actin-filaments": None,
         "Microtubules": None,
         "Endoplasmic-reticulum": None,
         "Nuclear-envelope": None
    }

    run_all_orgs(selected_model, best_orgs)

    # run_with_data_gen(dir="%s_%s" % (selected_model, organelle), model_name=selected_model, epochs=10, batch_size=64,
    #                   org_type=organelle, load_model_date=None, over_lap=1, multiply_img_z=1)

    # organelle = "Mitochondria"
    # run(dir="%s_%s" % (selected_model, organelle), model_name=selected_model, epochs=100, batch_size=32, read_img=True,
    #     org_type=organelle, img_read_limit=200, multiply_img_z=4)

