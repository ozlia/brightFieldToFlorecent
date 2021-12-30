from abc import ABC, abstractmethod


class ICNN(ABC):

    @abstractmethod
    def __init__(self, input_dim=(128, 128, 1), batch_size=16, epochs=1000):
        pass

    @abstractmethod
    def train(self, train_x, train_label, val_set=0.0, model_dir="/home/ozlia/basicAE/model2D/"):
        pass

    @abstractmethod
    def predict_patches(self, test_data_input):
        """

        @param test_data_input: list of bright_field patches
        @return: list of predicted fluorescent patches
        """
        pass

    @abstractmethod
    def predict(self, test_x):
        """
        predicts list of full scaled bright_field images
        @param test_x: list of full scaled predicted fluorescent images
        @return:
        """
        pass

    @abstractmethod
    def load_model(self, path='/home/ozlia/basicAE/model/'):
        """
        loads model
        @param path: path in cluster of saved model
        @return: nothing, saves the model in class
        """
