from __future__ import print_function, division

# import tensorflow.keras as keras
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.layers import Input, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split

from BasicAE import data_prepere


class Pix2Pix:
    def __init__(self):
        self.root_dir = '/home/tomrob/pix2pix'
        # self.progress_report : pd.DataFrame = pd.DataFrame(columns=['Epoch', 'Batch', 'G Loss', 'D Loss', 'D Acc'])
        self.progress_report = {k : [] for k in ['Epoch', 'Batch', 'G Loss', 'D Loss'] }
        # os.mkdir(self.root_dir)

        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # TODO get data
        self.org_type = "Mitochondria/"
        self.tiff_paths = data_prepere.load(self.org_type)
        self.tiffs_train, self.tiffs_test = train_test_split(self.tiff_paths, test_size=0.3, random_state=13)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)
        d5 = conv2d(d4, self.gf * 8)
        d6 = conv2d(d5, self.gf * 8)
        d7 = conv2d(d6, self.gf * 8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf * 8)
        u2 = deconv2d(u1, d5, self.gf * 8)
        u3 = deconv2d(u2, d4, self.gf * 8)
        u4 = deconv2d(u3, d3, self.gf * 4)
        u5 = deconv2d(u4, d2, self.gf * 2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=50, sample_interval_in_batches=50):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):  # TODO read once and save instead of reading every epoch
            for batch_i, (brightfield_batch, real_fluorescent_batch) in enumerate(
                    data_prepere.load_images_as_batches(brightfield_fluorescent_tiff_paths=self.tiffs_train,
                                                        batch_size=batch_size, img_res=(self.img_rows, self.img_cols))):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_fluorescent_batch = self.generator.predict(brightfield_batch)  # TODO predict>?

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([real_fluorescent_batch, brightfield_batch], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_fluorescent_batch, brightfield_batch], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([real_fluorescent_batch, brightfield_batch],
                                                      [valid, real_fluorescent_batch])

                self.document_progress(curr_epoch=epoch+1, total_epochs=epochs,curr_batch= batch_i,d_loss=d_loss,g_loss= g_loss,start_time=start_time)


                # If at save interval => save generated image samples
                if (batch_i + 1) % sample_interval_in_batches == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):

        images_root_dir = os.path.join(self.root_dir, 'images')
        # os.makedirs(images_root_dir, exist_ok=True)
        num_imgs = 3
        rows, cols = num_imgs, num_imgs
        brightfield, fluorescent = data_prepere.load_images_as_batches(
            brightfield_fluorescent_tiff_paths=self.tiffs_test,
            img_res=(self.img_rows, self.img_cols),
            sample_size=num_imgs).__next__()
        gen_fluorescent = self.generator.predict_on_batch(brightfield)
        gen_imgs = np.concatenate([brightfield, np.squeeze(gen_fluorescent), fluorescent])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['brightfield', 'gen fluorescent', ' real fluorescent']
        fig, axs = plt.subplots(rows, cols)
        cnt = 0
        for i in range(rows):
            for j in range(cols):
                axs[i, j].imshow(gen_imgs[cnt])
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
        os.makedirs(progress_root_dir,exist_ok=True)
        os.makedirs(models_root_dir,exist_ok=True)

        save_model(model=self.combined, filepath=os.path.join(models_root_dir, 'generator_model'))
        save_model(model=self.discriminator, filepath=os.path.join(models_root_dir, 'discriminator_model'))

        time = f"{datetime.datetime.now().strftime('%d-%m-%Y, %H:%M:%S')}.csv"

        progress_report : pd.DataFrame = pd.DataFrame(data=self.progress_report)
        progress_report.to_csv(path_or_buf=os.path.join(progress_root_dir,time))

    def load_model(self, target_path=None):
        if not target_path:
            target_path = os.path.join(self.root_dir, 'models')
        # try:
        self.combined = load_model(filepath=os.path.join(target_path, 'generator_model'))
        self.discriminator = load_model(filepath=os.path.join(target_path, 'discriminator_model'))
        # except:
        # raise FileNotFoundError(f'Could not load models from path: {target_path}')

    def document_progress(self, curr_epoch, total_epochs, curr_batch, d_loss, g_loss, start_time):
        elapsed_time = datetime.datetime.now() - start_time
        # Plot the progress
        for k,v in zip(self.progress_report.keys(),[curr_epoch,curr_batch, g_loss[0], d_loss[0]]):
            self.progress_report[k].append(v)

        # print("[Epoch %d/%d] [Batch %d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (curr_epoch, total_epochs,
        #                                                                                    curr_batch,
        #                                                                                    d_loss[0],
        #                                                                                    100 * d_loss[1],
        #                                                                                    g_loss[0],
        #                                                                                    elapsed_time))


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=10, batch_size=16, sample_interval_in_batches=3)
    gan.save_model_and_progress_report()
