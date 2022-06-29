from __future__ import print_function, division
from helpers import utils
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Dropout, Concatenate, BatchNormalization, LeakyReLU, UpSampling2D, Conv2D, \
    Activation, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from skimage.metrics import peak_signal_noise_ratio as peak_snr, structural_similarity as ssim, \
    mean_squared_error as mse

import datetime
import numpy as np
import os
import pandas as pd

from DataGeneration import DataGenPreparation, DataGenerator


class P2P_Discriminator():
    def __init__(self, batch_size, patch_size_channels_last, use_patches=False):
        self.input_size_channels_last = patch_size_channels_last
        self.optimizer = Adam(0.0002, 0.5)

        # self.channels used to be 1
        if use_patches:  # Calculate patch size of D (PatchGAN)
            patchGAN_patch_size = 2 ** 4
            self.patch_size = int(self.input_size_channels_last / patchGAN_patch_size)
            patch_arr_size = (batch_size, self.patch_size, self.patch_size, self.input_size_channels_last[2])
            self.loss_fn = tf.keras.losses.mean_squared_error #'mse'
        else:
            patch_arr_size = (batch_size, self.input_size_channels_last[2])
            self.loss_fn = tf.keras.losses.binary_crossentropy # 'binary_crossentropy'

        self.n_filters = 64
        self.build_model(use_patches)

    def add_conv_layer(self, layer_input, filters, f_size=4, bn=True, dropout_rate=0):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if dropout_rate > 0:
            d = Dropout(dropout_rate)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def build_model(self, use_patches):
        img_A = Input(shape=self.input_size_channels_last)
        img_B = Input(shape=self.input_size_channels_last)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = self.add_conv_layer(combined_imgs, self.n_filters)
        d2 = self.add_conv_layer(d1, self.n_filters * 2, dropout_rate=0.1)
        d3 = self.add_conv_layer(d2, self.n_filters * 4)
        d4 = self.add_conv_layer(d3, self.n_filters * 8)

        if use_patches:  # filter output
            # originally padding same
            validity = Conv2D(1, kernel_size=self.input_size_channels_last, strides=1, padding='same',
                              activation='sigmoid')(d4)
        else:
            validity = Flatten()(d4)
            validity = Dense(1)(validity)
            validity = Activation('sigmoid')(validity)

        self.model = Model([img_A, img_B], validity)


class P2P_Generator():
    def __init__(self, patch_size_channels_last):
        self.filters = 64
        self.input_size = patch_size_channels_last
        self.build_model()
        self.optimizer = Adam(0.0002, 0.5)
        self.loss_fn = tf.keras.losses.mean_absolute_error

    def add_conv_layer(self, layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same', activation=LeakyReLU())(layer_input)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)  # new epsilon addition
        return d

    def add_deconv_layer(self, layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation=LeakyReLU())(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        # u = LeakyReLU(alpha=0.2)(u)
        return u

    def build_model(self):
        # Image input
        d0 = Input(shape=self.input_size)

        # Downsampling
        d1 = self.add_conv_layer(d0, self.filters, bn=False)
        d2 = self.add_conv_layer(d1, self.filters * 2)
        d3 = self.add_conv_layer(d2, self.filters * 4)
        d4 = self.add_conv_layer(d3, self.filters * 8)
        d5 = self.add_conv_layer(d4, self.filters * 16)
        d6 = self.add_conv_layer(d5, self.filters * 16)

        # Upsampling
        u1 = self.add_deconv_layer(d6, d5, self.filters * 8, dropout_rate=0.1)
        u2 = self.add_deconv_layer(u1, d4, self.filters * 8, dropout_rate=0.1)
        u3 = self.add_deconv_layer(u2, d3, self.filters * 4)
        u4 = self.add_deconv_layer(u3, d2, self.filters * 2)
        u5 = self.add_deconv_layer(u4, d1, self.filters)

        u6 = UpSampling2D(size=2)(u5)
        output_img = Conv2D(self.input_size[2], kernel_size=4, strides=1, padding='same', activation='sigmoid')(u6)

        self.model = Model(d0, output_img)


class Pix2Pix(Model):
    def __init__(self, patch_size_channels_last, batch_size, print_summary=False, utilize_patchGAN=False):
        super(Pix2Pix, self).__init__()
        self.progress_report = {k: [] for k in ['Epoch', 'Batch', 'G Loss', 'D Loss']}
        self.utilize_patchGAN = utilize_patchGAN
        self.batch_size = batch_size
        self.patch_size_channels_last = patch_size_channels_last
        self.num_total_batches = 0

        self.full_gen = P2P_Generator(self.patch_size_channels_last)
        self.full_disc = P2P_Discriminator(batch_size=self.batch_size,patch_size_channels_last=patch_size_channels_last,use_patches=utilize_patchGAN)
        self.generator = self.full_gen.model
        self.discriminator = self.full_disc.model

        self.compile()

        if print_summary:
            self.print_summary()

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                **kwargs):

        super(Pix2Pix, self).compile()
        self.loss_weights = [0.5, 0.5]

        self.discriminator.compile(loss=self.full_disc.loss_fn, optimizer=self.full_disc.optimizer)
        self.generator.compile(loss=self.full_gen.loss_fn, optimizer=self.full_gen.optimizer)

    def call(self, inputs):
        return self.generator(inputs)

    def train_step(self, data):
        brightfield_img, real_floro_img = data
        print(brightfield_img.shape)

        # Decode them to fake images
        fake_floro_img = self.generator(brightfield_img)

        # Assemble labels discriminating real from fake images
        labels_shape = (1,) + self.patch_size_channels_last
        valid_labels = tf.ones(labels_shape) #+ 0.05 * tf.random.uniform(tf.shape(labels_shape))
        fake_labels = tf.zeros(labels_shape) #+ 0.05 * tf.random.uniform(tf.shape(labels_shape))

        if self.num_total_batches % 10 == 0:  # leaning towards training generator better
            # Train the discriminator
            with tf.GradientTape() as tape:
                real_predictions = self.discriminator([fake_floro_img, brightfield_img])
                d_loss_valid = self.full_disc.loss_fn(valid_labels, real_predictions)
            grads = tape.gradient(d_loss_valid, self.discriminator.trainable_weights)
            self.full_disc.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )

            with tf.GradientTape() as tape:
                fake_predictions = self.discriminator([real_floro_img, brightfield_img])
                d_loss_fake = self.full_disc.loss_fn(fake_labels, fake_predictions)
            grads = tape.gradient(d_loss_fake, self.discriminator.trainable_weights)
            self.full_disc.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )

            d_loss = (d_loss_valid + d_loss_fake) / 2
        else:
            d_loss = (0, 0)

        # Train the generator
        # Do not update the discriminator weights

        with tf.GradientTape() as tape:
            # Loss w.r.t ground truth
            gen_gt_loss = self.full_gen.loss_fn(real_floro_img, fake_floro_img)
            weighted_gen_losses = [self.loss_weights[0] * gen_gt_loss,self.loss_weights[1] * d_loss_valid]

        grads = tape.gradient(weighted_gen_losses, self.generator.trainable_weights)
        self.full_gen.optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        del tape

        self.num_total_batches += 1
        return {"d_loss": d_loss, "g_loss": 0}

    def sample_images(self, epoch, batch_i):
        brightfield, fluorescent = self.data_handler.load_images_as_batches(batch_size=1,
                                                                            sample_size=self.nImages_to_sample).__next__()
        fig_name = f'e{epoch}_b{batch_i}.png'
        utils.sample_images(self.generator, brightfield, fluorescent, fig_name, rescale=False,
                            org_type=self.data_handler.org_type)

    def save_model_and_progress_report(self, target_path=None):
        if target_path is None:
            target_path = os.path.join(utils.DIRECTORY, self.data_handler.org_type)
        models_root_dir = os.path.join(target_path, 'models')
        progress_root_dir = os.path.join(target_path, 'progress_reports')
        os.makedirs(progress_root_dir, exist_ok=True)
        os.mkdir(models_root_dir)

        save_model(model=self.combined, filepath=os.path.join(models_root_dir, 'combined_component'))
        save_model(model=self.generator, filepath=os.path.join(models_root_dir, 'generator_model'))
        save_model(model=self.discriminator, filepath=os.path.join(models_root_dir, 'discriminator_model'))

        fname = f"{datetime.datetime.now().strftime('%d-%m-%Y, %H:%M:%S')}.csv"
        progress_report: pd.DataFrame = pd.DataFrame.from_dict(data=self.progress_report)
        progress_report.to_csv(os.path.join(progress_root_dir, fname), index=False)

    def load_model(self, target_path=None, transfer_learning=False):
        if target_path is None:
            target_path = os.path.join(utils.DIRECTORY, self.data_handler.org_type, 'models')
        try:
            self.generator = load_model(filepath=os.path.join(target_path, 'generator_model'), compile=False)
            if transfer_learning:
                self.combined = load_model(filepath=os.path.join(target_path, 'combined_component'), compile=False)
                self.discriminator = load_model(filepath=os.path.join(target_path, 'discriminator_model'))
        except:
            raise FileNotFoundError(f'Could not load models from path: {target_path}')

    def document_progress(self, curr_epoch, total_epochs, curr_batch, d_loss, g_loss, start_time, sample_interval):
        elapsed_time = datetime.datetime.now() - start_time

        if curr_batch % sample_interval == 0:
            for k, v in zip(self.progress_report.keys(), [curr_epoch, curr_batch, g_loss[0], d_loss]):
                self.progress_report[k].append(v)

        print("[Epoch %d/%d] [Batch %d] [D loss: %f] [G loss: %f] time: %s" % (curr_epoch, total_epochs,
                                                                               curr_batch,
                                                                               d_loss,
                                                                               g_loss[0],
                                                                               elapsed_time))

    def predict_and_save(self):  # TODO test this
        root_dir = os.path.join(utils.DIRECTORY, self.data_handler.org_type)
        preds_dir_name = 'predicted_images'
        eval_metrics: dict = {k: [] for k in ['peak_snr', 'ssim', 'mse']}
        for i in range(len(self.data_handler.X_test)):
            curr_imgs_output_dir = os.path.join(self.data_handler.org_type, preds_dir_name, str(i))
            os.makedirs(name=os.path.join(utils.DIRECTORY, curr_imgs_output_dir),
                        exist_ok=True)  # stupid save requires this

            brightfield = np.expand_dims(self.data_handler.X_test[i], axis=0)  # for patchify process
            real_fluorescent = self.data_handler.y_test[i]
            gen_fluorescent = utils.predict_on_imgs(self.generator, brightfield,
                                                    self.data_handler.img_size_channels_first)

            # TODO might need to narrow down to only channel 2 of fluorescent
            eval_metrics['peak_snr'].append(
                peak_snr(real_fluorescent, gen_fluorescent, data_range=1.0))  # we need the maximal pixel value
            eval_metrics['ssim'].append(ssim(real_fluorescent, gen_fluorescent, data_range=1.0, channel_axis=2))
            eval_metrics['mse'].append(mse(real_fluorescent, gen_fluorescent))

            utils.save_full_2d_pic(gen_fluorescent[:, :, 2], os.path.join(curr_imgs_output_dir, 'gen_fluorescent.png'))
            utils.save_full_2d_pic(real_fluorescent[:, :, 2],
                                   os.path.join(curr_imgs_output_dir, 'real_fluorescent.png'))
            utils.save_full_2d_pic(brightfield[0][:, :, 2], os.path.join(curr_imgs_output_dir, 'brightfield.png'))

        fname = f"eval_{self.data_handler.org_type}.csv"
        pd.DataFrame.from_dict(data=eval_metrics).to_csv(os.path.join(root_dir, fname), index=False)

    def print_summary(self):
        self.generator.summary()
        self.discriminator.summary()
        # self.combined.summary()
        # models_dir = os.path.join(self.root_dir, 'pix2pix', 'models')
        # os.makedirs(models_dir, exist_ok=True)
        # plot_model(self.generator, to_file=os.path.join(models_dir, 'generator_model_plot.png'), show_shapes=True,
        #            show_layer_names=True)
        # plot_model(self.discriminator, to_file=os.path.join(models_dir, 'discriminator_model_plot.png'),
        #            show_shapes=True, show_layer_names=True)


if __name__ == '__main__':
    # Current changes   -   90x10 loss weights, leakyRelu all layers, extra gen layer, dropout on disc
    # TODO
    # 1. images should be sampled on epoch_end on DataGen with validation set
    # 2. progress should be documented with g loss and d loss every epoch using Keras CallBack
    # 3. if we want to train disc with different batch, not flexible and uneasy to accomplish
    # 4. Custom callback to sample images in end of batch - in CustomCallbacks file

    print_summary = True

    # training params
    batch_size = 32
    epochs = 300
    validation_size = 0.0
    test_size = 0.3
    batch_size = 32
    utilize_patchGAN = False
    # nImages_to_sample = 3 #TODO insert into DataGen

    # Data Parameters
    org_type = "Mitochondria"
    img_size = (640, 896, 6)
    patch_size = (128, 128, 6)
    # resplit = False

    gan = Pix2Pix(patch_size_channels_last=patch_size, batch_size=batch_size, print_summary=print_summary,
                  utilize_patchGAN=utilize_patchGAN)
    dgp = DataGenPreparation.DataGeneratorPreparation(img_size_channels_last=img_size,
                                                      patch_size_channels_last=patch_size, org_type=org_type,
                                                      resplit=False, validation_size=validation_size,
                                                      test_size=test_size)
    train_data_gen = DataGenerator.DataGeneratorDeprecatedRNG(patches_path=utils.get_dir(dgp.patches_dir_path),
                                                              batch_size=batch_size,
                                                              patch_size=patch_size, data_set='Train')

    if validation_size > 0:
        validation_data_gen = DataGenerator.DataGeneratorDeprecatedRNG(patches_path=utils.get_dir(dgp.patches_dir_path),
                                                                       batch_size=batch_size,
                                                                       patch_size=patch_size, data_set='Validation')
    else:
        validation_data_gen = None
    # TODO add callbacks
    # callbacks = [
    #     # keras.callbacks.ModelCheckpoint("%sBasicAEModel3D.h5" % model_dir, save_best_only=True),
    #     CSVLogger('log.csv' , append=True, separator=';')
    # ]

    # TODO: Currently only one img in test set because the second image had 4 channels and wasn't processed
    gan.fit(train_data_gen, validation_data=validation_data_gen, epochs=epochs, shuffle=True,
            verbose=1)
    # gan.predict_and_save()
    # gan.save_model_and_progress_report()
