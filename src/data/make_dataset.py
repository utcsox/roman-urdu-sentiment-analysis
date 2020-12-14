"""Module to load data"""
from typing import Any, List, Tuple

import emoji
import functools
import nltk
import numpy as np
import operator
import os
import pandas as pd
import re
import string

from sklearn.preprocessing import LabelEncoder
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_roman_urdu_sentiment_analysis_dataset(data_path: str, file_name: str, header_names: List[str],
                                               seed: int = 123, test_split: float = 0.2) -> Tuple[Any, Any]:
    """Loads the Roman Urdu sentiment analysis dataset
    Args:
        :param data_path: path to the data directory
        :param file_name: name of data file
        :param header_names: names of each column
        :param seed: random seed
        :param test_split:  percentage of data to use for test

    :return:
        A dataframe object with the texts and labels

    #References
        Sharf et al., 'Lexical normalization of roman Urdu text.' IJCSNS 17.12 (2017): 213.

    Download csv from:
    http://archive.ics.uci.edu/ml/datasets/Roman+Urdu+Data+Set
    """
    raw_df = _load_and_shuffle_data(data_path, file_name, header_names, seed)
    cleaned_df = _preprocess_raw_data(raw_df)
    cleaned_df['comment'] = cleaned_df['comment'].apply(lambda x: tokenizer(x))

    # Get the comment and sentiment labels
    texts = list(cleaned_df['comment'])
    labels = np.array(cleaned_df['sentiment'])

    return _split_training_and_test_sets(texts, labels, test_split)


def _load_and_shuffle_data(data_path: str, file_name: str, header_names: List[str], seed: int, encoding: str = 'utf-8',
                           skipinitialspace: bool = True) -> pd.DataFrame:
    """Loads and shuffle data using pandas
    Args:
        :param data_path: path to the data repository
        :param file_name: name of the data file
        :param header_names: names of each column
        :param seed: random seed
        :param encoding: encoding to use for UTF when reading/writing
        :param skipinitialspace: skip spaces after delimiter

    :return:
        A tuple of training and test data.
    """
    np.random.seed(seed)
    data_path = os.path.join(data_path, file_name)
    data = pd.read_csv(data_path, names=header_names, encoding=encoding, skipinitialspace=skipinitialspace)
    return data.reindex(np.random.permutation(data.index))


def _preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """ pre-process raw dataframe
    Args:
        :param df: raw dataframe

    :return:
        A cleaned dataframe that hae no misspelling and no nan values + encode target labels to numeric value
    """
    df_ = df.copy()
    df_.drop('nan', axis=1, inplace=True)
    df_.dropna(axis=0, subset=['comment'], inplace=True)
    df_.replace(to_replace='Neative', value='Negative', inplace=True)
    df_.dropna(subset=['sentiment'], inplace=True)

    le = LabelEncoder()
    le.fit(df_['sentiment'])
    df_['sentiment'] = le.transform(df_['sentiment'])

    return df_


def tokenizer(doc: pd.Series) -> str:
    """
    Tokenize a single document
    :param doc: one comment
    :return: a "clean" string that remove punctuation, numbers and split emoji apart
    """
    tokens = [word.lower() for word in nltk.word_tokenize(doc)]
    tokens = [re.sub(r'[0-9]', '', word) for word in tokens]
    tokens = [re.sub(r'[' + string.punctuation + ']', '', word) for word in tokens]
    tokens = ' '.join(tokens)
    em_split_emoji = emoji.get_emoji_regexp().split(tokens)
    em_split_whitespace = [substr.split() for substr in em_split_emoji]
    em_split = functools.reduce(operator.concat, em_split_whitespace)
    tokens = ' '.join(em_split)

    return tokens


def _split_training_and_test_sets(texts: pd.Series, labels: pd.Series, test_split: float):
    """Splits the texts and labels into training and test sets.
    # Arguments
        texts: text data.
        labels: label data.
        test_split: float, percentage of data to use for test.
    # Returns
        A tuple of training and test data.
    """
    num_training_samples = int((1 - test_split) * len(texts))
    return ((texts[:num_training_samples], labels[:num_training_samples]),
            (texts[num_training_samples:], labels[num_training_samples:]))


if __name__ == '__main__':
    DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
    ((train_comments, train_labels), (test_comments, test_labels)) = load_roman_urdu_sentiment_analysis_dataset(
                                                    DATA_DIR, 'Roman Urdu DataSet.csv', ['comment', 'sentiment', 'nan'])
    print(f'training has {len(train_comments)} examples, test has {len(test_comments)} examples.')
    print(f'train comments: {train_comments[0:5]}, train_labels: {train_labels[0:5]}')
