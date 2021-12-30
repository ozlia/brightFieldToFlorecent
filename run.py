import data_prepere
import keras
from sklearn.model_selection import train_test_split
import utils
from BasicAE.autoEncoder import AutoEncoder
from datetime import datetime

# interpreter_path = /home/omertag/.conda/envs/my_env/bin/python - change your user !!
# (x,y,z)
img_size = (6, 128, 128)
img_size_rev = (img_size[1], img_size[2], img_size[0])
batch_size = 128
epochs = 100
limit = 150
org_type = "Mitochondria/"

start = datetime.now()
print("reading images")
data_input, data_output = data_prepere.separate_data(data_prepere.load_paths(org_type, limit=limit), img_size)
data_input = utils.transform_dimensions(data_input, [0, 2, 3, 1])
data_output = utils.transform_dimensions(data_output, [0, 2, 3, 1])
utils.save_numpy_array(data_input, "input_images_after_data_prepare")
utils.save_numpy_array(data_output, "output_images_after_data_prepare.npy")
# data_input = utils.load_numpy_array("input_images_after_data_prepare.npy")
# data_output = utils.load_numpy_array("outpt_images_after_data_prepare.npy")
patches_input = utils.utils_patchify(data_input, img_size_rev, resize=True, over_lap_steps=1)
patches_output = utils.utils_patchify(data_output, img_size_rev, resize=True, over_lap_steps=1)

stop = datetime.now()
print('Done Reading and Patching, Time: ', stop - start)

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()
print("init model")
model = AutoEncoder(img_size_rev, epochs=epochs, batch_size=batch_size)
print("training model")
train_x, test_x, train_y, test_y = train_test_split(patches_input, patches_output, test_size=0.1, random_state=3,
                                                    shuffle=True)

model.train(train_x, train_y, val_set=0.1)
stop = datetime.now()
print('Done Train, Time: ', stop - start)

# model.load_model(model_dir="/model2D_full/")

print("Generate new pic")
br = model.predict([data_input[0]])
utils.save_full_2d_pic(br[:, :, 3], 'predicted_output.png')
utils.save_full_2d_pic(data_input[0][:, :, 3], 'input.png')
utils.save_full_2d_pic(data_output[0][:, :, 3], 'ground_truth.png')
print("All pics saved")
stop = datetime.now()
print('Done All, Time: ', stop - start)
