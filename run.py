import data_prepare
from sklearn.model_selection import train_test_split
import utils
from BasicAE.autoEncoder import AutoEncoder
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import backend as KB
# interpreter_path = /home/<username>/.conda/envs/<env name>/bin/python - change your user !!


img_size = (6, 64, 64)    # (x,y,z)
img_size_rev = (img_size[1], img_size[2], img_size[0])
batch_size = 32
epochs = 5000
limit = 150
org_type = "Nuclear-envelope/" # change the organelle name


start = datetime.now()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("reading images")

# data_input, data_output = data_prepare.separate_data(data_prepare.load_paths(org_type, limit=limit), img_size)
# utils.save_numpy_array(data_input, "input_images_after_data_prepare")
# utils.save_numpy_array(data_output, "output_images_after_data_prepare")

data_input = utils.load_numpy_array("input_images_after_data_prepare.npy")
data_output = utils.load_numpy_array("output_images_after_data_prepare.npy")
data_input = utils.transform_dimensions(data_input, [0, 2, 3, 1])
data_output = utils.transform_dimensions(data_output, [0, 2, 3, 1])
patches_input = utils.utils_patchify(data_input, img_size_rev, resize=True, over_lap_steps=2)
patches_output = utils.utils_patchify(data_output, img_size_rev, resize=True, over_lap_steps=2)

stop = datetime.now()
print('Done Reading and Patching, Time: ', stop - start)

# Free up RAM in case the model definition cells were run multiple times
KB.clear_session()
print("init model")
model = AutoEncoder(img_size_rev, epochs=epochs, batch_size=batch_size)
print("training model")
train_x, test_x, train_y, test_y = train_test_split(patches_input, patches_output, test_size=0.1, random_state=3,
                                                    shuffle=True)

model.train(train_x[: int(len(train_x)/2)], train_y[: int(len(train_y)/2)], val_set=0.1, model_dir="/AutoEncoder3D_64px/")
model.train(train_x[int(len(train_x)/2):], train_y[int(len(train_y)/2):], val_set=0.1, model_dir="/AutoEncoder3D_64px/")
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
