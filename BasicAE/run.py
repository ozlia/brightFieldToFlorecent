import data_prepere
import aearc
import keras
img_size = (128, 128)
num_classes = 3
train_number = 80


org_type = "Mitochondria/"
data_input, data_output = data_prepere.separate_data(data_prepere.load(org_type))

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = aearc.get_model(img_size)
model.summary()

train_data_input = data_input[:80]
train_data_output = data_output[:80]
test_data_input = data_input[80:]
test_data_output = data_output[80:]
callbacks = [
    keras.callbacks.ModelCheckpoint("BasicAEModel.h5", save_best_only=True)
]
model.fit(train_data_input, train_data_output, epochs=15, shuffle=True, callbacks=callbacks)
model.save("basicAE/model/")
