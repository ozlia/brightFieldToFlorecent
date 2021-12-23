import data_prepere
import keras
from sklearn.model_selection import train_test_split
import utils
from BasicAE.autoEncoder import AutoEncoder

# (x,y,z)
img_size = (128, 128, 1)
batch_size = 28
epochs = 100
limit = 150
org_type = "Mitochondria/"

print("reading images")
data_input, data_output = data_prepere.separate_data(data_prepere.load(org_type, limit=limit), img_size)
patches_input = utils.utils_patchify(data_input, img_size, resize=True)
patches_output = utils.utils_patchify(data_output, img_size, resize=True)
# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()
print("init model")
model = AutoEncoder(img_size, epochs=epochs)
print("training model")
train_x, test_x, y_train, y_test = train_test_split(patches_input, patches_output, test_size=0.1, random_state=3,
                                                    shuffle=True)
model.train(train_x, y_train)
# model.load_model(model_dir="/model2D/")

print("Generate predictions for n samples")
# print("shape text_x: " + str(test_x.shape))
# predictions = model.predict_patches(test_x)
# print("shape predictions: " + str(predictions.shape))

br = model.pre dict([data_input[0]])
utils.save_full_2d_pic(br)

# saves wanted patch
# utils.save_img(data_input[0], data_output[0], predictions[0])
# saves all the patches
# utils.save_entire_patch_series(data_input[:28], data_output[:28])
