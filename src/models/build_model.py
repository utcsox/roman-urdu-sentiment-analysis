"""Module to create model.

Methods to define multi-layer perceptron

"""
from typing import Any, List, Tuple

from tensorflow.keras import Sequential
from tensorflow.keras import Dense, Dropout


def mlp_model(layers: int, units: int, dropout_rate: float, input_shape: Tuple(int, int), num_classes: int):
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

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='softmax'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=num_classes, activation='sigmoid'))
    return model
