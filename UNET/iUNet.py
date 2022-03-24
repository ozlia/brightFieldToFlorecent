from abc import abstractmethod

from ICNN import ICNN


class iUNet(ICNN):
    @abstractmethod
    def get_down_sample_layer(self, layer_input, filters, f_size=(3, 3), max_pooling_stride=(2, 2), bn=False,
                              dropout_rate=0):
        pass

    @abstractmethod
    def get_up_sample_layer(self, layer_input=None, skip_input=None, filters=64, f_size=(3, 3), strides=(2, 2),
                            bn=False, dropout_rate=0):
        pass

    @abstractmethod
    def build_model(self, input_dim, base_filters, filter_size,
                    strides):  # can be divided to build_down_sampling, build_embedding,build_upsampling
        pass
