"""
@author : Shreyash Garg, Created on 02.08.2023

Initial Notes:
-This file is part of project investigating latent space and informative features.
-The file implements Autoencoder for MNIST digit dataset.
-The rest of the project will be built over this so I am
trying to implement every model from scratch and pushing to github.

The structure of network is not conventional because it is designed to be modular
so that I can add more functionality later.

"""

"""
Import packages
"""
import tensorflow as tf
import time

from tensorflow.keras.datasets import mnist

"""
Import user defined libraries
"""
from autoencoder_mnist_nn import Autoencoder, AE_loss_function



"""
tf function graph for a single training step
"""

@tf.function
def train_step(data_batch):
    with tf.GradientTape() as tape:
        # Perform training on one batch
        generated_data_batch = ae(data_batch, training=True)

        # get the latent units
        # latent = autoencoder.encoder(data_batch).numpy()

        # get the classified answer

        # compute the loss
        ae_loss = AE_loss_function(x_true=data_batch,
                                     x_gen=generated_data_batch,
                                     y_true=0,
                                     y_pred=0)
    # Compute gradients
    gradients = tape.gradient(ae_loss, ae.trainable_weights)

    # Perform one step of gradient descent
    optimizer.apply_gradients(zip(gradients, ae.trainable_weights))

    # return the loss for current batch
    return ae_loss



"""
Function for training
"""


def ae_train(train_data, epochs):

    # list to store epoch wise MSE loss
    epoch_loss = []
    for epoch in range(epochs):
        print("\nStart of epoch ", epoch, '')
        start_time = time.time()

        # Initialise list to store batchwise loss for a given epoch
        batchwise_loss = []
        for train_data_batch in train_data:
            batch_loss = train_step(train_data_batch)
            batchwise_loss.append(batch_loss)

        epoch_loss.append(sum(batchwise_loss) / batchsize)
        print("Time taken: ", (time.time() - start_time))

        print('\n Epoch training loss', epoch_loss[epoch])

    return epoch_loss



if __name__ == "__main__":

    """
    Prepare the dataset
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    data_shape = x_train.shape
    label_shape = y_train.shape

    # normalise the dataset between 0 and 1
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Reshaping because keras layers requires 4 dimensional tensor
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # data batch generator for training
    train_data = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(64)

    """
    Initialise variables and network parameters
    """
    # learning rate
    lr = 0.001

    # epochs
    epochs = 100

    # batchsize
    batchsize = 64

    # Initialise the optimiser
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Initialise the Autoencoder model
    ae = Autoencoder()

    # Run the autoencoder
    loss = ae_train(train_data,epochs)

