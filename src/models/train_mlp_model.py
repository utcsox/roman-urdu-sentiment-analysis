"""Module to train multi-layer perceptron model"""
from typing import Any, List, Tuple
from src.features.vectorize_data import vectorize
from src.models.build_model import mlp_model
from tensorflow.keras import callbacks

import argparse
import datetime
import tensorflow as tf

def train_mlp_model(data: Tuple[Any, Any], ):

    # load data
    (x_train, train_labels), (x_test, test_labels) = data

    # vectorize the text
    vectorize(x_train, train_labels, x_test)

    # load model
    model = mlp_model(layers=layers, units, dropout_rate, input_shape=input_shape, num_classes=num_classes)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=))

    # call_back function
    checkpoint_cb = callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1)
    tensorboard_cb = callbacks.TensorBoard(tensorboard_path, histogram_freq=1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
