import os
from IPython.display import Image, display
import PIL
from PIL import ImageOps
from tensorflow import keras
import UNET_ARC
from oxford_pets import OxfordPets
import numpy as np
from keras.preprocessing.image import load_img


input_dir = "images/"
target_dir = "annotations/trimaps/"
img_size = (160, 160)
num_classes = 3
batch_size = 32

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

# # Display input image #7
# Image(filename=input_img_paths[9]).show()
#
# # Display auto-contrast version of corresponding target (per-pixel categories)
# img = PIL.ImageOps.autocontrast(load_img(target_img_paths[9])).show()



# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = UNET_ARC.get_model(img_size, num_classes)
model.summary()
import random

# Split our img paths into a training and a validation set
val_samples = 32
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = OxfordPets(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 15
# model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
#
# model.save('path/to/location')
model = keras.models.load_model('oxford_segmentation.h5')

i = 15

val_preds = model.predict(val_gen)


def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    # display(img)
    img.show()


# Display results for validation image #10

# Display input image
with PIL.Image.open(val_input_img_paths[i]) as input:
    input.show()
# Display ground-truth target mask
img = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))

img.show()
# Display mask predicted by our model
display_mask(i)  # Note that the model only sees inputs at 150x150.
