import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import data_prepere
import aearc
import keras
from sklearn.model_selection import train_test_split


def save_img(data_input, data_output, predictions):
    Image.fromarray(np.squeeze(data_input)*255).convert('L').save('/home/ozlia/prediction/input.png')
    Image.fromarray(np.squeeze(predictions)*255).convert('L').save('/home/ozlia/prediction/prediction.png')
    Image.fromarray(np.squeeze(data_output)*255).convert('L').save('/home/ozlia/prediction/orginal.png')


def evaluate(model, test_data_input, test_data_output):
    # # # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(test_data_input, test_data_output, batch_size=batch_size)
    print("test loss, test acc:", results)


def train(data_input, data_output):
    # Build model
    model = aearc.get_model(img_size, color=num_classes)
    model.summary()

    train_data_input, test_data_input, train_data_output, test_data_output = train_test_split(data_input, data_output,
                                                                                              test_size=0.2,
                                                                                              random_state=13)
    train_X, valid_X, train_label, valid_label = train_test_split(train_data_input, train_data_output, test_size=0.2,
                                                                  random_state=13)

    callbacks = [
        keras.callbacks.ModelCheckpoint("/home/ozlia/BasicAEModel.h5", save_best_only=True)
    ]
    model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(valid_X, valid_label),
              callbacks=callbacks)
    model.save("/home/ozlia/basicAE/model/")


img_size = (128, 128)
num_classes = 1
batch_size = 16
epochs = 100

org_type = "Mitochondria/"
data_input, data_output = data_prepere.separate_data(data_prepere.load(org_type), img_size[0], img_size[1], num_classes)

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# train(data_input, data_output)
model = keras.models.load_model('/home/ozlia/basicAE/model/')

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 1 samples")
predictions = model.predict(data_input)

save_img(data_input[0], data_output[0], predictions[0])
