"""
@author : Shreyash Garg
Python file containing neural network
"""

""""
Import libraries
"""
import tensorflow as tf

from tensorflow.keras import layers,losses
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

"""
Autoencoder section for MNIST dataset

"""


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu',
                          padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu',
                          padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size=3, strides=2,
                                   activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2,
                                   activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3),
                          activation='sigmoid', padding='same')])


    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


"""
Loss function for supervised autoencoder
"""


def AE_loss_function(x_true, x_gen, y_true, y_pred):
    mse_loss = K.mean(K.square(x_true - x_gen))
    ae_loss = mse_loss
    return ae_loss
