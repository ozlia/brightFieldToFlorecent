from keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Dropout, concatenate, MaxPooling2D, Input
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import Model, models, callbacks
from patchify import unpatchify
from datetime import datetime
import utils
from UNET.iUNet import iUNet
import getpass
import os


class Unet(iUNet):

    def get_down_sample_layer(self, layer_input, filters, f_size=(3, 3), max_pooling_stride=(2, 2), bn=False,
                              dropout_rate=0):
        skip_connections_layer = Conv2D(filters=filters, kernel_size=f_size, padding='same', activation='relu')(
            layer_input)
        if dropout_rate > 0:
            skip_connections_layer = Dropout(dropout_rate)(skip_connections_layer)
        skip_connections_layer = Conv2D(filters=filters, kernel_size=f_size, padding='same', activation='relu')(
            skip_connections_layer)
        if max_pooling_stride is None:
            down_sample_layer = skip_connections_layer
            skip_connections_layer = None
        else:
            down_sample_layer = MaxPooling2D(max_pooling_stride)(skip_connections_layer)
        return skip_connections_layer, down_sample_layer


    def get_up_sample_layer(self, layer_input=None, skip_input=None, filters=64, f_size=(3, 3), strides=(2, 2),
                            bn=False, dropout_rate=0):
        upsample_layer = Conv2DTranspose(filters, kernel_size=(2, 2), strides=strides, padding='same')(layer_input)
        upsample_layer = concatenate([upsample_layer, skip_input])

        # activation = LeakyRelU
        upsample_layer = Conv2D(filters=filters, kernel_size=f_size, padding='same', activation='relu')(upsample_layer)
        if dropout_rate > 0:
            upsample_layer = Dropout(dropout_rate)(upsample_layer)
        upsample_layer = Conv2D(filters=filters, kernel_size=f_size, padding='same', activation='relu')(upsample_layer)
        return upsample_layer

    def build_model(self, input_dim, base_filters, filter_size, strides):
        inputs = Input(shape=input_dim)
        # encoder
        c1, d1 = self.get_down_sample_layer(inputs, base_filters, filter_size, strides, dropout_rate=0.1)
        c2, d2 = self.get_down_sample_layer(d1, base_filters * 2, filter_size, strides, dropout_rate=0.1)
        c3, d3 = self.get_down_sample_layer(d2, base_filters * 4, filter_size, strides, dropout_rate=0.2)
        c4, d4 = self.get_down_sample_layer(d3, base_filters * 8, filter_size, strides, dropout_rate=0.2)

        # embedding
        d5,_ = self.get_down_sample_layer(d4, base_filters * 16, filter_size, max_pooling_stride=None,
                                           dropout_rate=0.3)
        # decoder
        u1 = self.get_up_sample_layer(layer_input=d5, skip_input=c4, filters=base_filters * 8, f_size=filter_size,
                                      strides=strides)
        u2 = self.get_up_sample_layer(layer_input=u1, skip_input=c3, filters=base_filters * 4, f_size=filter_size,
                                      strides=strides)
        u3 = self.get_up_sample_layer(layer_input=u2, skip_input=c2, filters=base_filters * 2, f_size=filter_size,
                                      strides=strides)
        u4 = self.get_up_sample_layer(layer_input=u3, skip_input=c1, filters=base_filters, f_size=filter_size,
                                      strides=strides)

        outputs = Conv2D(input_dim[2], filter_size, padding='same', activation='sigmoid', name='decoder_output')(u4)

        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mae')  # optimizer = Adam(0.0002, 0.5)
        return model

    def __init__(self, input_dim=(128, 128, 6), batch_size=32, epochs=1000):

        base_filters = 16
        filter_size = (3, 3)
        strides = (2, 2)  # max sampling strides in deconv, strides in upsampling

        self.model = self.build_model(input_dim, base_filters, filter_size, strides)
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs

        self.USER = getpass.getuser().split("@")[0]
        print(self.USER)

        self.dir = "/home/%s/%s" % (self.USER, "basicAE")
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def train(self, train_x, train_label, val_set=0.0, model_dir="model3D_full"):
        save_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        model_dir = "%s/%s_%s/" % (self.dir, model_dir, save_time)
        # train_x = utils.transform_dimensions(train_x, [0, 2, 3, 1])
        # train_label = utils.transform_dimensions(train_label, [0, 2, 3, 1])
        # model_dir = self.dir + model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model = self.model
        model.summary()
        callback = [
            callbacks.ModelCheckpoint("%sBasicAEModel3D.h5" % model_dir, save_best_only=True),
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

    def predict(self, img):
        """
        predicts list of full scaled bright_field images
        @param path_to_tiff: list of full scaled predicted fluorescent images
        @return:
        """
        bright_field = utils.utils_patchify(img, self.input_dim)
        for row in bright_field:
            for col in row:
                pred_img = self.model.predict(col)
                col[0] = pred_img[0]
        size = img[0].shape
        return unpatchify(bright_field, size)

    def load_model(self, model_dir='/model/'):
        """
        loads model
        @param model_dir: path in cluster of saved model
        @return: nothing, saves the model in class
        """
        path = self.dir + model_dir
        self.model = models.load_model(path)
