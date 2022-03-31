import numpy as np
import data_prepare
from sklearn.model_selection import train_test_split
import utils
from BasicAE.autoEncoder import AutoEncoder
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import backend as KB
import pandas as pd
from pandas import DataFrame

# interpreter_path = /home/<username>/.conda/envs/<env name>/bin/python - change your user !!

METADATA_CSV_PATH = "/sise/assafzar-group/assafzar/fovs/metadata.csv"
img_size = (6, 64, 64)    # (x,y,z)
# batch_size = 32
# epochs = 1000
# org_type = "Mitochondria/"  # change the organelle name

def run(epochs, batch_size, dir, read_img = False, org_type = None, img_read_limit = 150 ):
    img_size_rev = (img_size[1], img_size[2], img_size[0])
    start = datetime.now()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("reading images")

    if read_img:
        data_input, data_output = data_prepare.separate_data(data_prepare.load_paths(org_type, limit=img_read_limit), img_size)
        utils.save_numpy_array(data_input, "input_images_after_data_prepare_norm")
        utils.save_numpy_array(data_output, "output_images_after_data_prepare_norm")
    else:
        data_input = utils.load_numpy_array("input_images_after_data_prepare_norm.npy")
        data_output = utils.load_numpy_array("output_images_after_data_prepare_norm.npy")

    data_input = utils.transform_dimensions(data_input, [0, 2, 3, 1])
    data_output = utils.transform_dimensions(data_output, [0, 2, 3, 1])
    patches_input = utils.utils_patchify(data_input, img_size_rev, resize=True, over_lap_steps=1)
    patches_output = utils.utils_patchify(data_output, img_size_rev, resize=True, over_lap_steps=1)

    stop = datetime.now()
    print('Done Reading and Patching, Time: ', stop - start)

    # Free up RAM in case the model definition cells were run multiple times
    KB.clear_session()
    print("init model")
    model = AutoEncoder(img_size_rev, epochs=epochs, batch_size=batch_size)
    print("training model")
    train_x, test_x, train_y, test_y = train_test_split(patches_input, patches_output, test_size=0.1, random_state=3,
                                                        shuffle=True)
    # split_to = 4
    # split_x = np.array_split(train_x, split_to)
    # split_y = np.array_split(train_y, split_to)
    # for i in range(split_to):
    #     model.train(split_x[i], split_y[i], val_set=0.1, model_dir="/AutoEncoder3D_64px/")

    model.train(train_x[: int(len(train_x)/2)], train_y[: int(len(train_y)/2)], val_set=0.1, model_dir=dir)
    model.train(train_x[int(len(train_x)/2):], train_y[int(len(train_y)/2):], val_set=0.1, model_dir=dir)
    stop = datetime.now()
    print('Done Train, Time: ', stop - start)

    # model.load_model(model_dir="/model2D_full/")

    print("Generate new pic")
    save_time = datetime.now().strftime("%H-%M_%d-%m-%Y")
    predicted_img = model.predict([data_input[0]])
    print("Saving .........")
    utils.save_np_as_tiff(predicted_img, save_time, "predict")
    utils.save_np_as_tiff(data_input[0], save_time, "input")
    utils.save_np_as_tiff(data_output[0], save_time, "ground_truth")
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


def cmd_script():
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

    # run(epochs= epochs, batch_size= batch_size, dir=utils.DIR_NAME_INPUT, read_img=read_img, org_type=org, img_read_limit=150)

    # print_full(matadata_df)
    # print_full((matadata_df.head()))
    # ['StructureDisplayName']
    # ['ChannelNumberBrightfield']

if __name__ == '__main__':
    cmd_script()