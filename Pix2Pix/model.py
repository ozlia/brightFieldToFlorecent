from __future__ import print_function, division
import utils
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Dropout, Concatenate, BatchNormalization, LeakyReLU, UpSampling2D, Conv2D, \
    Activation, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from skimage.metrics import peak_signal_noise_ratio as peak_snr, structural_similarity as ssim, \
    mean_squared_error as mse

import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from Pix2Pix.data_handler import data_handler


class Pix2Pix:
    def __init__(self, batch_size=-1, print_summary=False, utilize_patchGAN=True, nImages_to_sample=3):
        self.progress_report = {k: [] for k in ['Epoch', 'Batch', 'G Loss', 'D Loss']}
        self.utilize_patchGAN = utilize_patchGAN
        self.nImages_to_sample = nImages_to_sample

        # Input shape
        self.data_handler = data_handler()
        self.img_shape = self.data_handler.img_size_rev  # 128x128x6
        self.img_rows = self.img_shape[0]
        self.img_cols = self.img_shape[1]
        self.channels = self.img_shape[2]

        if self.utilize_patchGAN:
            # Calculate patch size of D (PatchGAN)
            patchGAN_patch_size = 2 ** 4
            patch = int(self.img_shape[1] / patchGAN_patch_size)
            self.disc_patch = (patch, patch, 1)
            disc_loss = 'mse'
        else:
            disc_loss = 'binary_crossentropy'

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        self.build_model(disc_loss)

        if print_summary:
            self.print_summary()

    def build_model(self, disc_loss):
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

        if self.utilize_patchGAN: # filter output
            validity = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(d4)
        else:
            validity = Flatten()(d4)
            validity = Dense(1)(validity)
            validity = Activation('sigmoid')(validity)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size_in_patches=50, sample_interval_in_batches=50,
              report_sample_interval_in_batches=1,shuffle_batches=False):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        if self.utilize_patchGAN:
            patch_arr_size = (batch_size_in_patches,) + self.disc_patch
        else:
            patch_arr_size = (batch_size_in_patches, 1)

        valid = np.ones(patch_arr_size)
        fake = np.zeros(patch_arr_size)

        d_loss = (0, 0)

        for epoch in range(epochs):
            for batch_i, (real_brightfield_batch, real_fluorescent_batch) in enumerate(
                    self.data_handler.load_images_as_batches(batch_size=batch_size_in_patches,shuffle=shuffle_batches)):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                fake_fluorescent_batch = self.generator.predict(real_brightfield_batch)

                # Train the discriminators (original images = real / generated = Fake)

                if batch_i % 10 == 0:  # leaning towards training generator better
                    # disc_real_brightfield_batch, disc_real_fluorescent_batch = batch_generator.__next__()
                    # batch_i += 1
                    self.discriminator.trainable = True
                    d_loss_real = self.discriminator.train_on_batch([real_fluorescent_batch, real_brightfield_batch],
                                                                    valid)
                    d_loss_fake = self.discriminator.train_on_batch([fake_fluorescent_batch, real_brightfield_batch],
                                                                    fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                self.discriminator.trainable = False
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
            gen_fluorescent = utils.patchify_predict_imgs(self.generator, brightfield, self.data_handler.img_size_rev)

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
    # addition to running offline?
    # os.system("nohup bash -c '" +
    #           sys.executable + " pix2pix.py --size 192 >result.txt" +
    #           "' &")

    batch_size = 32  # 3500 patches, ~107 batches per epoch if limit=150
    img_sample_interval_in_batches = 53
    report_sample_interval_in_batches = 32
    print_summary = False
    utilize_patchGAN = False
    nImages_to_sample = 3
    shuffle_batches = True

    gan = Pix2Pix(print_summary=print_summary, utilize_patchGAN=utilize_patchGAN, nImages_to_sample=nImages_to_sample)
    gan.train(epochs=100, batch_size_in_patches=batch_size, sample_interval_in_batches=img_sample_interval_in_batches,
              report_sample_interval_in_batches=report_sample_interval_in_batches,shuffle_batches=shuffle_batches)
    gan.predict_and_save()
    gan.save_model_and_progress_report()
