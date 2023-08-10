"""
@author : Shreyash Garg, Created on 05.08.2023

Initial Notes:
-This file is part of project investigating latent space and informative features.
-The file implements Supervised Autoencoder for MNIST digit dataset.
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

from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist

"""
Import user defined libraries
"""
from SupervisedAE_MNIST_nn import SupervisedAE


"""
tf function graph for a single training step
"""


@tf.function
def train_step(data_batch, labels_batch):
    with tf.GradientTape() as tape:
        # Perform training on one batch
        generated_data_batch = sae(data_batch, training=True)

        # get the latent units
        latent = sae.encoder(data_batch)

        # get the classified answer
        generated_labels_batch = sae.classifier(latent, training = False)

        # compute the loss
        sae_loss = sae_loss_function(x_true=data_batch,
                                     x_gen=generated_data_batch,
                                     y_true=labels_batch,
                                     y_pred=generated_labels_batch)
    # Compute gradients
    gradients = tape.gradient(sae_loss, sae.trainable_weights)

    # Perform one step of gradient descent
    optimizer.apply_gradients(zip(gradients, sae.trainable_weights))

    # return the loss for current batch
    return sae_loss


"""
Function for training
"""


def sae_train(train_dataset, epochs):

    # list to store epoch wise MSE loss
    epoch_loss = []
    for epoch in range(epochs):
        print("\nStart of epoch ", epoch, '')
        start_time = time.time()

        # Initialise list to store batchwise loss for a given epoch
        batchwise_loss = []
        for train_data_batch, train_labels_batch in train_dataset:
            batch_loss = train_step(train_data_batch, train_labels_batch)
            batchwise_loss.append(batch_loss)

        epoch_loss.append(sum(batchwise_loss) / batchsize)
        print("Time taken: ", (time.time() - start_time))

        print('\n Epoch training loss', epoch_loss[epoch])

    return epoch_loss


"""
Loss function for supervised autoencoder
"""


def sae_loss_function(x_true, x_gen, y_true, y_pred):
    mse_loss = K.mean(K.square(x_true - x_gen))
    classification_loss = classifier_loss_fn(y_true, y_pred)
    sae_loss = mse_loss + classification_loss
    return sae_loss


"""
Main function
"""


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

    # one hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

    # data batch generator for training
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(64)

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

    # initailise loss function for classification
    classifier_loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # Initialise the Autoencoder model
    sae = SupervisedAE()

    # Run the autoencoder
    loss = sae_train(train_dataset, epochs)
