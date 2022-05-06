import numpy as np
import data_prepare
from sklearn.model_selection import train_test_split
import utils
from Img2ImgAE.autoEncoder import AutoEncoder
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import backend as KB
import pandas as pd
from pandas import DataFrame
from argparse import ArgumentParser
from CrossDomainAE.crossDomainAE import AutoEncoderCrossDomain
from UNET.SpecialUnetLiad import Unet

# interpreter_path = /home/<username>/.conda/envs/<env name>/bin/python - change your user !!
# interpreter_path_omer  = /home/omertag/.conda/envs/my_env/bin/python

METADATA_CSV_PATH = "/sise/assafzar-group/assafzar/fovs/metadata.csv"
img_size = (6, 64, 64)  # (x,y,z)


# batch_size = 32
# epochs = 1000
# org_type = "Mitochondria/"  # change the organelle name

def run(dir, model_name, epochs=1000, batch_size=32, read_img=False, org_type=None, img_read_limit=150):
    utils.set_dir(dir)
    img_size_rev = (img_size[1], img_size[2], img_size[0])
    start = datetime.now()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("reading images")

    if org_type:
        if not org_type[-1] == "/":
            org_type = org_type + "/"

    if read_img:
        data_input, data_output = data_prepare.separate_data(data_prepare.load_paths(org_type, limit=img_read_limit),
                                                             img_size)
        utils.save_numpy_array_as_dict(data_input, "input_images_after_data_prepare_norm")
        utils.save_numpy_array_as_dict(data_output, "output_images_after_data_prepare_norm")
        print("Saved successfully numpy array at %s" % utils.DIRECTORY)
    else:
        data_input = utils.load_numpy_array("input_images_after_data_prepare_norm.npy")
        data_output = utils.load_numpy_array("output_images_after_data_prepare_norm.npy")
        print("Loaded successfully numpy array from %s" % utils.DIRECTORY)

    data_input = utils.transform_dimensions(data_input, [0, 2, 3, 1])
    data_output = utils.transform_dimensions(data_output, [0, 2, 3, 1])
    patches_input = utils.utils_patchify(data_input, img_size_rev, resize=True, over_lap_steps=1)
    patches_output = utils.utils_patchify(data_output, img_size_rev, resize=True, over_lap_steps=1)

    stop = datetime.now()
    print('Done Reading and Patching, Time: ', stop - start)

    # Free up RAM in case the model definition cells were run multiple times
    KB.clear_session()
    print("init model")
    model = create_model(model_name, img_size_rev=img_size_rev, epochs=epochs, batch_size=batch_size)
    print("training model")
    train_x, test_x, train_y, test_y = train_test_split(patches_input, patches_output, test_size=0.1, random_state=3,
                                                        shuffle=True)
    # split_to = 4
    # split_x = np.array_split(train_x, split_to)
    # split_y = np.array_split(train_y, split_to)
    # for i in range(split_to):
    #     model.train(split_x[i], split_y[i], val_set=0.1, model_dir="/AutoEncoder3D_64px/")

    model.train(train_x, train_y, val_set=0.1, model_dir=dir)
    # model.train(train_x, train_y, val_set=0.1, model_dir=dir)
    stop = datetime.now()
    print('Done Train, Time: ', stop - start)

    # model.load_model(model_dir="/model2D_full/")

    print("Generate new pic")
    save_time = datetime.now().strftime("%H-%M_%d-%m-%Y")
    predicted_img = model.predict([data_input[0]])
    print("Saving .........")
    utils.save_np_as_tiff(predicted_img, save_time, "predict", model)
    utils.save_np_as_tiff(data_input[0], save_time, "input", model)
    utils.save_np_as_tiff(data_output[0], save_time, "ground_truth", model)
    # utils.save_full_2d_pic(predicted_img[:, :, 2], 'predicted_output_32px.png')
    # utils.save_full_2d_pic(data_input[0][:, :, 2], 'input.png')
    # utils.save_full_2d_pic(data_output[0][:, :, 2], 'ground_truth.png')
    print("... All tiffs saved !!")
    stop = datetime.now()
    print('Done All, Time: ', stop - start)


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

    run(dir=dir_name, epochs=epochs, batch_size=batch_size, read_img=read_img, org_type=org, img_read_limit=150)

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


def create_model(name: str, img_size_rev, epochs, batch_size):
    name = name.lower()
    case = {
        "img2img": AutoEncoder(img_size_rev, epochs=epochs, batch_size=batch_size),
        "crossdomain": AutoEncoderCrossDomain(img_size_rev, epochs=epochs, batch_size=batch_size),
        "unet": Unet(img_size_rev),  # todo liad : change to your config
        "pix2pix": None
    }
    return case.get(name, None)


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
    run(model_name=args.model_type, epochs=args.epochs, batch_size=args.batch_size, dir=args.dir,
        read_img=args.read_img, org_type=args.org_type, img_read_limit=args.read_limit)


if __name__ == '__main__':
    # run(epochs=200, batch_size=16, dir="BasicAE_64x_2*2", read_img=True, org_type="Mitochondria",
    #     img_read_limit=200)
    # run(dir="BasicAE_64x_2*2", epochs=200, batch_size=32)
    parse_command_line()
