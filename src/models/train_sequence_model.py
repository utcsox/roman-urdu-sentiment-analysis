""" Module to train sequence model"""
from typing import Any, Tuple
from src.data.make_dataset import load_roman_urdu_sentiment_analysis_dataset
from src.features.vectorize_data import tf_keras_vectorize
from src.models.build_model import sequence_model

import argparse
import datetime
import tensorflow as tf

TOP_K = 20000


def train_sequence_model(data: Tuple[Any, Any], embedding_dim: int = 200, dropout_rate: float = 0.2,
                         filters: int = 3, kernel_size: int = 3, pool_size: int = 3, num_classes: int = 3,
                         learning_rate: float = 1e-3, epochs: int = 1000, batch_size: int = 128):
    """
    Train sequence model of the dataset
    :param data: A tuple of training/test data
    :param embedding_dim:  dimension of the embedding vectors
    :param dropout_rate: percentage of input to drop at Droput Layers
    :param filters: output dimension of the layers
    :param kernel_size: length of the convolution window
    :param pool_size: factor by which to downscale input at MaxPooling layer
    :param num_classes: number of output classes
    :param learning_rate: learning rate of training model
    :param epochs: number of epochs
    :param batch_size: number of samples per batch
    """
    (train_text, train_labels), (test_text, test_labels) = data
    x_train, x_test, word_index = tf_keras_vectorize(train_text, test_text)

    num_features = len(word_index) + 1
    model = sequence_model(num_features=num_features, embedding_dim=embedding_dim, input_shape=x_train.shape[1],
                           dropout_rate=dropout_rate, filters=filters, kernel_size=kernel_size, pool_size=pool_size,
                           num_classes=num_classes)

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

    # create callback for early stopping on validation loss.  If the loss does not decrease in 5 consecutive tries,
    # end training loops
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]

    # train & validate models
    history = model.fit(x_train, train_labels, epochs=epochs, callbacks=callbacks,
                        validation_data=(x_test, test_labels),
                        verbose=2, batch_size=batch_size)

    history = history.history
    print(f'Test accuracy: {history["val_acc"][-1]}, loss:{history["val_loss"][-1]}')

    # Save model
    model.save('sequence_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]


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
        default=['comment', 'sentiment', 'nan'],
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
    test_accuracy, test_loss = train_sequence_model(dataset)
