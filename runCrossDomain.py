import numpy as np
import data_prepare
from sklearn.model_selection import train_test_split
import utils
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import backend as KB
# interpreter_path = /home/<username>/.conda/envs/<env name>/bin/python - change your user !!
from CrossDomainAE.crossDomainAE import AutoEncoderCrossDomain

img_size = (6, 64, 64)    # (x,y,z) patch size
img_size_rev = (img_size[1], img_size[2], img_size[0])
batch_size = 32
epochs = 100
limit = 150  # how many TIFF you would like to run
org_type = "Mitochondria/"  # change the organelle name


start = datetime.now()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("reading images")

# run only once - if you change the amount of tiffs you need to run it again
# data_input, data_output = data_prepare.separate_data(data_prepare.load_paths(org_type, limit=limit), PATCH_SIZE)
# utils.save_numpy_array(data_input, "input_images_after_data_prepare_norm")
# utils.save_numpy_array(data_output, "output_images_after_data_prepare_norm")

data_input = utils.load_numpy_array("input_images_after_data_prepare_norm.npy")
data_output = utils.load_numpy_array("output_images_after_data_prepare_norm.npy")
data_input = utils.transform_dimensions(data_input, [0, 2, 3, 1])
data_output = utils.transform_dimensions(data_output, [0, 2, 3, 1])
# into two groups - test and train and then send only the train set split it and then send to patchify. then delete all the 34-35 lines the patchify.
train_x, test_x, train_y, test_y = train_test_split(data_input, data_output, test_size=0.1, random_state=3,
                                                    shuffle=True)  #  NEW
patches_train_x = utils.utils_patchify(train_x, img_size_rev, resize=True, over_lap_steps=1)
patches_train_y = utils.utils_patchify(train_y, img_size_rev, resize=True, over_lap_steps=1)

# patches_input = utils.utils_patchify(data_input, img_size_rev, resize=True, over_lap_steps=1)
# patches_output = utils.utils_patchify(data_output, img_size_rev, resize=True, over_lap_steps=1)

stop = datetime.now()
print('Done Reading and Patching, Time: ', stop - start)

# Free up RAM in case the model definition cells were run multiple times
KB.clear_session()
print("init model")
model = AutoEncoderCrossDomain(img_size_rev, epochs=epochs, batch_size=batch_size)
print("training model")
# train_x, test_x, train_y, test_y = train_test_split(patches_input, patches_output, test_size=0.1, random_state=3,
#                                                     shuffle=True)
# split_to = 4
# split_x = np.array_split(train_x, split_to)
# split_y = np.array_split(train_y, split_to)
# for i in range(split_to):
#     model.train(split_x[i], split_y[i], val_set=0.1, model_dir="/AutoEncoder3D_64px/")

model.train(patches_train_y, patches_train_y, val_set=0.1, model_dir="AutoEncoderF2F_64px")
stop = datetime.now()
print('Done Train, Time: ', stop - start)

# model.load_model(model_dir="/AutoEncoderB2B_64px_28-03-2022_20-29")

print("Generate new pic")
save_time = datetime.now().strftime("%H-%M_%d-%m-%Y")
predicted_img = model.predict([test_y[0]])  # replace test_x to data_output[0]
print("Saving .........")
utils.save_np_as_tiff(predicted_img, save_time, "predict")
# utils.save_np_as_tiff(test_x[0], save_time, "input")  # also here, test_x ->
utils.save_np_as_tiff(test_y[0], save_time, "ground_truth")  # also here, test_y ->
# utils.save_full_2d_pic(predicted_img[:, :, 2], 'predicted_output_32px.png')
# utils.save_full_2d_pic(data_input[0][:, :, 2], 'input.png')
# utils.save_full_2d_pic(data_output[0][:, :, 2], 'ground_truth.png')
print("... All tiffs saved !!")
stop = datetime.now()
print('Done All, Time: ', stop - start)
