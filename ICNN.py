import os
from abc import ABC, abstractmethod
from datetime import datetime
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import callbacks
from patchify import unpatchify
import utils
from tensorflow.keras import models

from smooth_tiled_predictions import predict_img_with_smooth_windowing


class ICNN(ABC):

    @abstractmethod
    def __init__(self, input_dim=(128, 128, 1), batch_size=16, epochs=1000):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.dir = None
        self.saved_model_dir = None

    def train(self, train_x, train_label, val_set=0.0, model_dir="/home/ozlia/basicAE/model2D/"):
        save_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        model_dir = "%s/%s_%s/" % (self.dir, model_dir, save_time)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model = self.model
        model.summary()
        callback = [
            callbacks.ModelCheckpoint("%sUnetModel3D.h5" % model_dir, save_best_only=True),
            CSVLogger('%slog_%s.csv' % (model_dir, save_time), append=True, separator=';')
        ]
        model.fit(train_x, train_label, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                  validation_split=val_set,
                  callbacks=callback)
        model.save(model_dir)

    def predict_patches(self, test_data_input):
        """
        @param test_data_input: list of bright_field patches
        @return: list of predicted fluorescent patches
        """
        return self.model.predict(test_data_input)

    def predict(self, test_x):
        """
        predicts list of full scaled bright_field images
        @param test_x: list of full scaled predicted fluorescent images
        @return:
        """
        bright_field = utils.utils_patchify(test_x, self.input_dim)
        for row in bright_field:
            for col in row:
                pred_img = self.model.predict(col)
                col[0] = pred_img[0]
        size = test_x[0].shape
        return unpatchify(bright_field, size)

    def load_model(self, model_dir='/home/ozlia/basicAE/model/'):
        """
        loads model
        @param model_dir: path in cluster of saved model
        @return: nothing, saves the model in class
        """
        path = self.dir + model_dir
        self.model = models.load_model(path)
        print("Loaded successfully from %s" % path)

    def predict_smooth(self, img):
        smooth_predicted_img = predict_img_with_smooth_windowing(
            img[0],
            self.input_dim[0],
            subdivisions=2,
            nb_classes=self.input_dim[2],
            pred_func=lambda img_batch_subdiv: self.predict_patches(img_batch_subdiv)
        )
        return smooth_predicted_img
