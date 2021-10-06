import data_prepere
import AE_ARC
import keras
img_size = (160, 160)
num_classes = 3


org_type = "Mitochondria/"
data_input, data_output = data_prepere.separate_data(data_prepere.load(org_type))

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = AE_ARC.get_model(img_size, num_classes)
model.summary()
