from helpers import utils


class PerformancePlotCallback(keras.callbacks.Callback):
    def __init__(self, model, validation_data_gen, org_type):
        self.model = model
        self.data_gen = validation_data_gen
        self.num_samples = 3
        self.org_type = org_type

    def on_epoch_end(self, epoch, logs=None):
        brightfield_batch, fluorescent_batch = self.data_gen.__getitem__()
        utils.sample_images(self.model, brightfield_batch[:self.num_samples], fluorescent_batch[:self.num_samples],
                            self.num_samples, rescale=False, org_type=self.org_type)
