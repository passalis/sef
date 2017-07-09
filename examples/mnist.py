# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt

import numpy as np
from keras.datasets import mnist


def load_mnist():
    """
    Loads the MNIST dataset
    :return:
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0
    x_test = x_test.reshape((x_test.shape[0], -1)) / 255.0

    return np.float32(x_train), y_train, np.float32(x_test), y_test
