import data_prepere
import keras
from sklearn.model_selection import train_test_split

import utils
from BasicAE.autoEncoder import AutoEncoder

# (x,y,z)
img_size = (128, 128, 1)
batch_size = 28
epochs = 100
#hello
org_type = "Mitochondria/"

print("reading images")
data_input, data_output = data_prepere.separate_data(data_prepere.load(org_type), img_size)

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()
print("init model")
model = AutoEncoder(img_size, epochs=epochs)
print("training model")
train_x, test_x, y_train, y_test = train_test_split(data_input, data_output, test_size=0.25, random_state=3,
                                                    shuffle=True)
model.train(train_x, y_train)
print("Generate predictions for n samples")
predictions = model.predict_patches(test_x)

# saves wanted patch
utils.save_img(data_input[0], data_output[0], predictions[0])
# saves all the patches
utils.save_entire_patch_series(data_input[:28], data_output[:28])
