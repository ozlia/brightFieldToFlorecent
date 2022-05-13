from __future__ import print_function, division

import data_prepare
import utils
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Dropout, Concatenate, BatchNormalization, LeakyReLU, UpSampling2D, Conv2D, \
    Activation, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from skimage.metrics import peak_signal_noise_ratio as peak_snr, structural_similarity as ssim, \
    mean_squared_error as mse
import sys
import datetime
import numpy as np
import os
import pandas as pd

from DataGeneration.DataGenPreparation.BasicDataGenPreparation import BasicDataGeneratorPreparation
from DataGeneration.DataGenerator.DataGenerator import DataGenerator


class P2P_Discriminator():
    def __init__(self, batch_size, patch_size_channels_last, use_patchGAN=False):
        self.n_filters = 64
        self.input_size_channels_last = patch_size_channels_last

        # Adversarial loss ground truths
        if use_patchGAN:  # Calculate patch size of D (PatchGAN)
            patchGAN_patch_size = 2 ** 4
            self.patch_size = int(self.input_size_channels_last[0] / patchGAN_patch_size)
            self.disc_patch = (self.patch_size, self.patch_size, self.input_size_channels_last[2])
            patch_arr_size = (batch_size,) + self.disc_patch
            self.loss = 'mse'
        else:
            patch_arr_size = (batch_size, 1)
            self.loss = 'binary_crossentropy'

        self.valid = np.ones(patch_arr_size)
        self.fake = np.zeros(patch_arr_size)

        self.build_model(use_patchGAN)

        self.optimizer = Adam(0.0002, 0.5)
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

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
            validity = Conv2D(1, kernel_size=self.patch_size, strides=1, padding='same', activation='sigmoid')(d4)
        else:
            validity = Flatten()(d4)
            validity = Dense(1)(validity)
            validity = Activation('sigmoid')(validity)

        self.model = Model([img_A, img_B], validity)


class P2P_Generator():
    def __init__(self, patch_size_channels_last):
        self.n_filters = 64
        self.input_size_channels_last = patch_size_channels_last
        self.build_model()

        self.loss = 'mae'
        self.optimizer = Adam(0.0002, 0.5)
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

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
        d0 = Input(shape=self.input_size_channels_last)

        # Downsampling
        d1 = self.add_conv_layer(d0, self.n_filters, bn=False)
        d2 = self.add_conv_layer(d1, self.n_filters * 2)
        d3 = self.add_conv_layer(d2, self.n_filters * 4)
        d4 = self.add_conv_layer(d3, self.n_filters * 8)
        d5 = self.add_conv_layer(d4, self.n_filters * 16)
        d6 = self.add_conv_layer(d5, self.n_filters * 16)

        # Upsampling
        u1 = self.add_deconv_layer(d6, d5, self.n_filters * 8, dropout_rate=0.1)
        u2 = self.add_deconv_layer(u1, d4, self.n_filters * 8, dropout_rate=0.1)
        u3 = self.add_deconv_layer(u2, d3, self.n_filters * 4)
        u4 = self.add_deconv_layer(u3, d2, self.n_filters * 2)
        u5 = self.add_deconv_layer(u4, d1, self.n_filters)

        u6 = UpSampling2D(size=2)(u5)
        output_img = Conv2D(self.input_size_channels_last[2], kernel_size=4, strides=1, padding='same',
                            activation='sigmoid')(u6)

        self.model = Model(d0, output_img)


class Pix2Pix:
    def __init__(self, patch_size_channels_last, batch_size, print_summary=False, utilize_patchGAN=True,
                 nImages_to_sample=3):
        self.progress_report = {k: [] for k in ['Epoch', 'Batch', 'G Loss', 'D Loss']}
        self.utilize_patchGAN = utilize_patchGAN
        self.num_images_in_sample = nImages_to_sample
        self.batch_size = batch_size

        # Input shape
        self.patch_size_channels_last = patch_size_channels_last

        if print_summary:
            self.print_summary()

        self.build_model()

    def build_model(self):
        self.full_disc = P2P_Discriminator(batch_size=self.batch_size,
                                           patch_size_channels_last=self.patch_size_channels_last,
                                           use_patchGAN=self.utilize_patchGAN)
        self.full_gen = P2P_Generator(self.patch_size_channels_last)
        self.discriminator = self.full_disc.model
        self.generator = self.full_gen.model

        # Input images and their conditioning images
        real_fluorescent = Input(shape=self.patch_size_channels_last)
        real_brightfield = Input(shape=self.patch_size_channels_last)

        # By conditioning on B generate a fake version of A
        fake_fluorescent = self.generator(real_brightfield)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_fluorescent, real_brightfield])

        self.combined = Model(inputs=[real_fluorescent, real_brightfield], outputs=[valid, fake_fluorescent])
        self.combined.compile(loss=[self.full_disc.loss, self.full_gen.loss], loss_weights=[10, 90],
                              optimizer=self.full_gen.optimizer)

    def custom_train_on_batch(self, epochs, data_gen: DataGenerator, save_target_dir):
        self.num_batches_in_epoch = len(data_gen)
        start_time = datetime.datetime.now()

        d_loss = (0, 0)
        for epoch in range(epochs):
            for batch_num in range(self.num_batches_in_epoch):
                brightfield_batch, real_fluorescent_batch = data_gen[batch_num]
                fake_fluorescent_batch = self.generator.predict(brightfield_batch)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                if batch_num % 10 == 0:  # leaning towards training generator better
                    # disc_real_brightfield_batch, disc_real_fluorescent_batch = batch_generator.__next__()
                    # batch_i += 1
                    d_loss_real = self.discriminator.train_on_batch([real_fluorescent_batch, brightfield_batch],
                                                                    self.full_disc.valid)
                    d_loss_fake = self.discriminator.train_on_batch([fake_fluorescent_batch, brightfield_batch],
                                                                    self.full_disc.fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                g_loss = self.combined.train_on_batch([real_fluorescent_batch, brightfield_batch],
                                                      [self.full_disc.valid, real_fluorescent_batch])

                self.document_progress(curr_epoch=epoch + 1, total_epochs=epochs, curr_batch=batch_num, d_loss=d_loss,
                                       g_loss=g_loss, start_time=start_time)
            fig_name = f'e{epoch}_b{batch_num}.png'
            utils.sample_images(model=self.generator, brightfield_patches=brightfield_batch[:self.num_images_in_sample],
                                fluorescent_patches=real_fluorescent_batch[:self.num_images_in_sample],
                                fig_name=fig_name,
                                rescale=False, target_dir=save_target_dir)

            data_gen.on_epoch_end()

    def save_model_and_progress_report(self, target_path):
        target_path = utils.get_dir(target_path)
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
        self.generator = load_model(filepath=os.path.join(target_path, 'generator_model'), compile=False)
        if transfer_learning:
            self.combined = load_model(filepath=os.path.join(target_path, 'combined_component'), compile=False)
            self.discriminator = load_model(filepath=os.path.join(target_path, 'discriminator_model'))

    def document_progress(self, curr_epoch, total_epochs, curr_batch, d_loss, g_loss, start_time):
        elapsed_time = datetime.datetime.now() - start_time

        if curr_batch == self.num_batches_in_epoch - 1:
            for k, v in zip(self.progress_report.keys(), [curr_epoch, curr_batch, g_loss[0], d_loss]):
                self.progress_report[k].append(v)

        print("[Epoch %d/%d] [Batch %d] [D loss: %f] [G loss: %f] time: %s" % (curr_epoch, total_epochs,
                                                                               curr_batch,
                                                                               d_loss,
                                                                               g_loss[0],
                                                                               elapsed_time))

    def predict_and_save_eval(self, test_data_gen: DataGenerator, img_size_channels_last, target_dir):
        preds_dir_name = f"Predictions_{datetime.datetime.now().strftime('%d-%m-%Y, %H:%M:%S')}"
        root_dir = os.path.join(target_dir, preds_dir_name)
        os.makedirs(utils.get_dir(root_dir), exist_ok=True)
        eval_metrics: dict = {k: [] for k in ['ssim', 'pearson']}

        for img_i in range(len(test_data_gen)):  # len is number of patches
            curr_img_name = test_data_gen.brightfield_patches_paths[img_i * test_data_gen.num_patches_in_img]
            # full_path/ImgName_i.npy -> ImgName
            curr_img_name = os.path.splitext(os.path.basename(curr_img_name))[0].rsplit('_', 1)[0]

            img_brightfield_patches, real_fluorescent_img = test_data_gen[img_i]
            fake_fluorescent_img_channels_last = self.predict_on_patches(model=self.generator,
                                                                         patches=img_brightfield_patches,
                                                                         img_size_channels_last=img_size_channels_last)

            curr_imgs_output_dir = os.path.join(root_dir, curr_img_name)
            os.makedirs(curr_imgs_output_dir, exist_ok=True)

            # TODO: It might be possible to use the Keras evaluation api and get some additional info
            eval_metrics['ssim'].append(
                ssim(real_fluorescent_img, fake_fluorescent_img_channels_last, data_range=1.0, channel_axis=2))
            eval_metrics['pearson'].append(
                np.corrcoef(real_fluorescent_img.flatten(), fake_fluorescent_img_channels_last.flatten())[0][1])

            print(f'Saving image number {img_i}')
            utils.save_np_as_tiff_v2(img_channels_last=fake_fluorescent_img_channels_last,
                                     fname=f'{curr_img_name}_fake',
                                     target_path=curr_imgs_output_dir)
            utils.save_np_as_tiff_v2(img_channels_last=real_fluorescent_img, fname=f'{curr_img_name}_real',
                                     target_path=curr_imgs_output_dir)
            utils.save_full_2d_pic(fake_fluorescent_img_channels_last[:, :, 0],
                                   os.path.join(curr_imgs_output_dir, f'{curr_img_name}_fake.png'))
            utils.save_full_2d_pic(real_fluorescent_img[:, :, 0],
                                   os.path.join(curr_imgs_output_dir, f'{curr_img_name}_real.png'))
        print(f'Saving eval metrics')
        pd.DataFrame.from_dict(data=eval_metrics).to_csv(utils.get_dir(root_dir), index=False)

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

    def predict_on_patches(self, model, patches: np.array,
                           img_size_channels_last):  # assuming img dims are (1,patch_dims)
        fake_fluorescent_patches = []
        for patch in patches:
            patch = np.expand_dims(patch, axis=0)
            fake_fluorescent_patches.append(model.predict(patch))
        # TODO needs to be calculated somehow from original patch size - fitted for 128x128
        fake_fluorescent_patches = np.reshape(np.array(fake_fluorescent_patches), newshape=(5, 7, 1,) + patches[
            0].shape)  # (35,patch_size) -> ((5, 7, 1, patch_size)
        fake_fluorescent_img_channels_last = utils.unpatchify(fake_fluorescent_patches, img_size_channels_last)
        # fake_fluorescent_img_channels_first = np.moveaxis(fake_fluorescent_img_channels_last, -1, 0)
        return fake_fluorescent_img_channels_last


if __name__ == '__main__':
    # Current changes   -   50x50 loss weights, leakyRelu all layers, extra gen layer, dropout on disc,

    first_time_testing_if_works = False
    print_summary = False

    # training params
    batch_size = 32
    epochs = 1
    validation_size = 0.0
    test_size = 0.3
    utilize_patchGAN = False
    # nImages_to_sample = 3 #TODO insert into DataGen

    # Data Parameters
    org_type = "Mitochondria"
    img_size_channels_first = (6, 640, 896)
    img_size_channels_last = (img_size_channels_first[1], img_size_channels_first[2], img_size_channels_first[0])
    patch_size_channels_last = (128, 128, 6)
    num_patches_in_img = img_size_channels_first[1] // patch_size_channels_last[0] * img_size_channels_first[2] // \
                         patch_size_channels_last[1]
    # resplit = False

    gan = Pix2Pix(patch_size_channels_last=patch_size_channels_last, batch_size=batch_size, print_summary=print_summary,
                  utilize_patchGAN=utilize_patchGAN)

    dgp = BasicDataGeneratorPreparation(img_size_channels_first=img_size_channels_first,
                                        patch_size_channels_last=patch_size_channels_last, org_type=org_type,
                                        resplit=False, validation_size=validation_size,
                                        test_size=test_size, initial_testing=first_time_testing_if_works)
    train_data_gen = DataGenerator(data_root_path=utils.get_dir(org_type), num_epochs=epochs,
                                   batch_size=batch_size, num_patches_in_img=num_patches_in_img
                                   , data_set_type='Train')
    test_data_gen = DataGenerator(data_root_path=utils.get_dir(org_type), num_epochs=epochs,
                                  batch_size=batch_size, num_patches_in_img=num_patches_in_img
                                  , data_set_type='Test')
    # if validation_size > 0:
    #     validation_data_gen = DataGenerator(patches_path=utils.get_dir(dgp.patches_dir_path),
    #                                                       batch_size=batch_size,
    #                                                       patch_size=patch_size, data_set='Validation')
    # else:
    #     validation_data_gen = None

    output_path = os.path.join(org_type, 'Output')

    gan.custom_train_on_batch(epochs=epochs, data_gen=train_data_gen, save_target_dir=output_path)
    gan.predict_and_save_eval(test_data_gen=test_data_gen, target_dir=output_path,
                              img_size_channels_last=img_size_channels_last)
    gan.save_model_and_progress_report(target_path=output_path)
