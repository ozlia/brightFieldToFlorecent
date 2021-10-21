import data_prepere
import aearc
import keras
from sklearn.model_selection import train_test_split

img_size = (128, 128)
num_classes = 3
batch_size = 16
epochs = 100

org_type = "Mitochondria/"
data_input, data_output = data_prepere.separate_data(data_prepere.load(org_type), img_size[0], img_size[1], num_classes)

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = aearc.get_model(img_size)
model.summary()

train_number = int(len(data_input)*(3/4))

train_data_input = data_input[:train_number]
train_data_output = data_output[:train_number]
test_data_input = data_input[train_number:]
test_data_output = data_output[train_number:]

train_X,valid_X,train_label,valid_label = train_test_split(train_data_input, train_data_output, test_size=0.2, random_state=13)

callbacks = [
    keras.callbacks.ModelCheckpoint("BasicAEModel.h5", save_best_only=True)
]
# model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs, verbose=1, validation_data=(valid_X, valid_label), callbacks=callbacks)
# model.save("basicAE/model/")

model = keras.models.load_model('BasicAEModel.h5')
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_data_input, test_data_output, batch_size=batch_size)
print("test loss, test acc:", results)

