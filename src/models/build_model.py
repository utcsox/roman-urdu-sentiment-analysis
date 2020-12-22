"""Module to create model.

Methods to define multi-layer perceptron

"""
from typing import Any, List, Tuple

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Embedding, MaxPool1D, GlobalAveragePooling1D)


def mlp_model(layers: int, units: int, dropout_rate: float, input_shape: Tuple[int, int], num_classes: int):
    """ Create an instance of multi-layer perceptron model.

    :param layers: number of densely-connected NN layers
    :param units: output dimensionality of the output space of a layer
    :param dropout_rate: the rate of a given layer to drop of from the network
    :param input_shape: shape of the input to model
    :param num_classes: number of output classes
    :return: a tensorflow.python.keras.engine.sequential.Sequential class of multi-layer perceptron model
    """

    model = Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers - 1):
        model.add(Dense(units=units, activation='softmax'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=num_classes, activation='sigmoid'))
    return model


def sequence_model(num_features: int, embedding_dim: int, input_shape: Tuple, dropout_rate: float, filters: int,
                   kernel_size: int, pool_size: int, num_classes: int):
    """Create an instance of CNN model
    :param num_features: embedding input dimension (# of words)
    :param embedding_dim: dimension of the embedding vectors
    :param input_shape: shape of input to the model
    :param dropout_rate: percentage of input to drop at Droput Layers
    :param filters: output dimension of the layers
    :param kernel_size: length of the convolution window
    :param pool_size: factor by which to downscale input at MaxPooling layer
    :param num_classes: number of output classes
    """

    model = Sequential()
    # Add Embedding layer.
    model.add(Embedding(input_dim=num_features,
                        output_dim=embedding_dim,
                        input_length=input_shape[0]
                        ))
    model.add(Dropout(rate=dropout_rate))
    model.add(Conv1D(filters=filters,
                     kernel_size=kernel_size,
                     activation='relu',
                     bias_initializer='random_uniform',
                     padding='same'))
    model.add(MaxPool1D(pool_size=pool_size))
    model.add(Conv1D(filters=filters * 2,
                     kernel_size=kernel_size,
                     activation='relu',
                     bias_initializer='random_uniform',
                     padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=num_classes, activation='softmax'))
    return model
