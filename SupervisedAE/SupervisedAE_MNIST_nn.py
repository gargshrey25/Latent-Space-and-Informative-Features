"""
@author : Shreyash Garg  Created on 05.08.2023
Python file containing neural network
"""

""""
Import libraries
"""
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model

"""
Autoencoder section for MNIST dataset

"""
"""
Supervised Autoencoder section for MNIST dataset

"""


class SupervisedAE(Model):
    def __init__(self):
        super(SupervisedAE, self).__init__()

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

        self.classifier = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(10, activation='softmax')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def classifier(self, latent):
        classified = self.classifier(latent)
        return classified
