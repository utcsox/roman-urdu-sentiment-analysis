"""Module to train multi-layer perceptron model"""

from typing import Any, Tuple
from src.data.make_dataset import load_roman_urdu_sentiment_analysis_dataset
from src.features.vectorize_data import vectorize
from src.models.build_model import mlp_model
from tensorflow.keras import callbacks


import argparse
import datetime
import tensorflow as tf


def train_mlp_model(data: Tuple[Any, Any], output_dir: str, layers: int = 2, units: int = 2, dropout_rate: float = 0.3,
                    num_classes: int = 3) -> Tuple[float, float]:
    """
    Train n-gram models for the roman urdu dataset

    :param data: A tuple of training/test data
    :param output_dir:  The output directory to save the model artifact
    :param layers: output dimensionality of the output space of a layer
    :param units: dimensionality of the output space of a layer
    :param dropout_rate: the rate of a given layer to drop of from the network
    :param num_classes: number of output classes
    :return:
    """
    # load data
    (x_train, train_labels), (x_test, test_labels) = data

    # vectorize the text
    x_train, x_test = vectorize(x_train, train_labels, x_test)

    # load model
    model = mlp_model(layers, units, dropout_rate, input_shape=x_train.shape[1:], num_classes=num_classes)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    # call_back function
    early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', patience=5)
    # checkpoint_cb = callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1)
    # tensorboard_cb = callbacks.TensorBoard(tensorboard_path, histogram_freq=1)

    print(model.summary())

    history = model.fit(
        x_train,
        train_labels,
        epochs=1000,
        callbacks=[early_stopping_cb],
        validation_data=(x_test, test_labels),
        verbose=2,
        batch_size=64)

    result = history.history
    print(f'Test accuracy: {result["val_acc"][-1]}, loss: {result["val_loss"][-1]}')
    model.save(output_dir + f"layers_{layers}_units_{units}_mlp_model.h5")

    return result["val_acc"][-1], result["val_loss"][-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../../data/raw/',
        help='input data directory'
    )
    parser.add_argument(
        '--file_name',
        type=str,
        default='Roman Urdu DataSet.csv',
        help='input file name'
    )
    parser.add_argument(
        '--header_names',
        default= ['comment', 'sentiment', 'nan'],
        help='headers of columns of interest'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../../models/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_',
        help='input data directory'
    )

    args, unparsed = parser.parse_known_args()
    dataset = load_roman_urdu_sentiment_analysis_dataset(args.data_dir, args.file_name, args.header_names)
    train_mlp_model(dataset, output_dir=args.output_dir)
