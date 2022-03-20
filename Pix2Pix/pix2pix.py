from __future__ import print_function, division

import sys
from tensorflow.keras.utils import plot_model
from patchify import unpatchify

import utils

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Dropout, Concatenate, BatchNormalization, LeakyReLU, UpSampling2D, Conv2D, \
    Activation, Flatten, Dense
from tensorflow.keras.optimizers import Adam
# from Pix2Pix.custom_metrics import wasserstein_loss

import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from Pix2Pix.pix_data_prepere import pix2pix_data_prepare


class Pix2Pix:
    def __init__(self, batch_size=-1, print_summary=False, utilize_patchGAN=True):
        self.root_dir = '/home/tomrob/pix2pix'
        os.makedirs(self.root_dir, exist_ok=True)
        self.progress_report = {k: [] for k in ['Epoch', 'Batch', 'G Loss', 'D Loss']}
        self.utilize_patchGAN = utilize_patchGAN

        # Input shape
        self.data_preper = pix2pix_data_prepare()
        self.img_shape = self.data_preper.img_size_rev
        self.img_rows = self.img_shape[0]
        self.img_cols = self.img_shape[1]
        self.channels = self.img_shape[2]

        if self.utilize_patchGAN:
            disc_loss = 'binary_crossentropy'
        else:
            # Calculate patch size of D (PatchGAN)
            patchGAN_patch_size = 2 ** 4
            patch = int(self.img_shape[1] / patchGAN_patch_size)
            self.disc_patch = (patch, patch, 1)
            disc_loss = 'mse'

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        self.build_model(disc_loss)

        if print_summary:
            self.print_summary()

    def build_model(self,disc_loss):
        self.d_optimizer = Adam(0.0002, 0.5)
        self.g_optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=disc_loss, optimizer=self.d_optimizer)

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        real_fluorescent = Input(shape=self.img_shape)
        real_brightfield = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_fluorescent = self.generator(real_brightfield)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_fluorescent, real_brightfield])

        self.combined = Model(inputs=[real_fluorescent, real_brightfield], outputs=[valid, fake_fluorescent])
        self.combined.compile(loss=[disc_loss, 'mae'], loss_weights=[1, 100], optimizer=self.g_optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same', activation='relu')(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)  # new epsilon addition
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            u = LeakyReLU(alpha=0.2)(u)
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)
        d5 = conv2d(d4, self.gf * 16)

        # Upsampling
        u2 = deconv2d(d5, d4, self.gf * 8, dropout_rate=0.1)
        u3 = deconv2d(u2, d3, self.gf * 4, dropout_rate=0.2)
        u4 = deconv2d(u3, d2, self.gf * 2)
        u5 = deconv2d(u4, d1, self.gf)

        u6 = UpSampling2D(size=2)(u5)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u6)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True, dropout_rate=0):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if dropout_rate > 0:
                d = Dropout(dropout_rate)(d)
            if bn:
                d = BatchNormalization(momentum=0.8, epsilon=0.01)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        if self.utilize_patchGAN:
            validity = Flatten()(d4)
            validity = Dense(1)(validity)
            validity = Activation('sigmoid')(validity)
        else:  # filter output
            validity = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size_in_patches=50, sample_interval_in_batches=50,
              report_sample_interval_in_batches=1):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        if self.utilize_patchGAN:
            patch_arr_size = (batch_size_in_patches, 1)
        else:
            patch_arr_size = (batch_size_in_patches,) + self.disc_patch

        valid = np.ones(patch_arr_size)
        fake = np.zeros(patch_arr_size)

        d_loss = (0, 0)

        for epoch in range(epochs):
            for batch_i, (real_brightfield_batch, real_fluorescent_batch) in enumerate(
                    self.data_preper.load_images_as_batches(batch_size=batch_size_in_patches)):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                fake_fluorescent_batch = self.generator.predict(real_brightfield_batch)

                # Train the discriminators (original images = real / generated = Fake)

                if batch_i % 10 == 0:  # leaning towards training generator better
                    # disc_real_brightfield_batch, disc_real_fluorescent_batch = batch_generator.__next__()
                    # batch_i += 1

                    d_loss_real = self.discriminator.train_on_batch([real_fluorescent_batch, real_brightfield_batch],
                                                                    valid)
                    d_loss_fake = self.discriminator.train_on_batch([fake_fluorescent_batch, real_brightfield_batch],
                                                                    fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([real_fluorescent_batch, real_brightfield_batch],
                                                      [valid, real_fluorescent_batch])

                self.document_progress(curr_epoch=epoch + 1, total_epochs=epochs, curr_batch=batch_i, d_loss=d_loss,
                                       g_loss=g_loss, start_time=start_time,
                                       sample_interval=report_sample_interval_in_batches)

                # If at save interval => save generated image samples
                if ((batch_i + 1) % sample_interval_in_batches) == 0:
                    self.sample_images(epoch, batch_i + 1)

                if d_loss < 0.001:  # Typically points towards vanishing gradient
                    raise InterruptedError('dloss was too low so generator has probably stopped learning at this point')

    def sample_images(self, epoch, batch_i):

        images_root_dir = os.path.join(self.root_dir, 'images')
        os.makedirs(images_root_dir, exist_ok=True)
        rows, cols = 3, 3  # num imgs over 3
        brightfield, fluorescent = self.data_preper.load_images_as_batches(batch_size=1, sample_size=3).__next__()
        gen_fluorescent = self.generator.predict(brightfield)
        gen_imgs = np.concatenate(
            [brightfield[:, :, :, 0], np.squeeze(gen_fluorescent[:, :, :, 0]), fluorescent[:, :, :, 0]])

        # TODO Rescale images 0 - 1 not sure if necessary
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['brightfield', 'gen fluorescent', 'real fluorescent']
        fig, axs = plt.subplots(rows, cols)
        cnt = 0
        for i in range(rows):
            for j in range(cols):
                axs[i, j].imshow(gen_imgs[cnt], cmap='gray')
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1
        fig_path = os.path.join(images_root_dir, f'e{epoch}_b{batch_i}.png')
        fig.savefig(fig_path)
        # plt.show()
        plt.close()

    def save_model_and_progress_report(self, target_path=None):
        if not target_path:
            target_path = os.path.join(self.root_dir)
        models_root_dir = os.path.join(target_path, 'models')
        progress_root_dir = os.path.join(target_path, 'progress_reports')
        os.makedirs(progress_root_dir, exist_ok=True)
        os.makedirs(models_root_dir, exist_ok=True)

        save_model(model=self.combined, filepath=os.path.join(models_root_dir, 'combined_component'))
        save_model(model=self.generator, filepath=os.path.join(models_root_dir, 'generator_model'))
        save_model(model=self.discriminator, filepath=os.path.join(models_root_dir, 'discriminator_model'))

        fname = f"{datetime.datetime.now().strftime('%d-%m-%Y, %H:%M:%S')}.csv"

        progress_report: pd.DataFrame = pd.DataFrame.from_dict(data=self.progress_report)
        progress_report.to_csv(os.path.join(progress_root_dir, fname), index=False)

    def load_model(self, target_path=None):
        if target_path is None:
            target_path = os.path.join(self.root_dir, 'models')
        # try:
        # self.combined = load_model(filepath=os.path.join(target_path, 'combined_component'),compile=False)
        self.generator = load_model(filepath=os.path.join(target_path, 'generator_model'), compile=False)

        # self.discriminator = load_model(filepath=os.path.join(target_path, 'discriminator_model'))
        # except:
        # raise FileNotFoundError(f'Could not load models from path: {target_path}')

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

    def load_model_predict_and_save(self):  # first image only
        self.load_model()
        data_input = utils.load_numpy_array(self.data_preper.input_img_array_path)
        data_output = utils.load_numpy_array(self.data_preper.output_img_array_path)
        data_input = data_input[:2]
        data_output = data_output[:2]
        data_input = utils.transform_dimensions(data_input, [0, 2, 3, 1])
        data_output = utils.transform_dimensions(data_output, [0, 2, 3, 1])

        img = [data_input[0]]
        bright_field = utils.utils_patchify(img, self.data_preper.img_size_rev)
        for row in bright_field:
            for col in row:
                pred_img = self.generator.predict(col)
                col[0] = pred_img[0]
        size = img[0].shape
        br = unpatchify(bright_field, size)
        utils.save_full_2d_pic(br[:, :, 2], 'predicted_output.png')
        utils.save_full_2d_pic(data_input[0][:, :, 2], 'input.png')
        utils.save_full_2d_pic(data_output[0][:, :, 2], 'ground_truth.png')

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
    # addition to running offline?
    # os.system("nohup bash -c '" +
    #           sys.executable + " pix2pix.py --size 192 >result.txt" +
    #           "' &")

    batch_size = 32
    print_summary = True
    img_sample_interval_in_batches = 53
    report_sample_interval_in_batches = 32
    utilize_patchGAN = True

    gan = Pix2Pix(print_summary=print_summary, utilize_patchGAN=utilize_patchGAN)
    gan.train(epochs=100, batch_size_in_patches=batch_size, sample_interval_in_batches=img_sample_interval_in_batches,
              report_sample_interval_in_batches=report_sample_interval_in_batches)
    gan.save_model_and_progress_report()
    # gan.load_model_predict_and_save()
