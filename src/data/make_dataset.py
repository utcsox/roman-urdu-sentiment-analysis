"""Module to load data"""
from typing import Any, List

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_roman_urdu_sentiment_analysis_dataset(data_path: str, file_name: str, header_names: List[str],
                                               seed: int = 123):
    """Loads the Roman Urdu sentiment analysis dataset
    Args:
        :param data_path: path to the data directory
        :param file_name: name of data file
        :param header_names: names of each column
        :param seed: random seed

    :return:
        A dataframe object with the texts and labels

    #References
        Sharf et al., 'Lexical normalization of roman Urdu text.' IJCSNS 17.12 (2017): 213.

    Download csv from:
    http://archive.ics.uci.edu/ml/datasets/Roman+Urdu+Data+Set
    """
    raw_df = _load_and_shuffle_data(data_path, file_name, header_names, seed)
    cleaned_df = _preprocess_raw_data(raw_df)


    return cleaned_df


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
        A dataframe object with the texts and labels
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


if __name__ == '__main__':
    DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
    df = load_roman_urdu_sentiment_analysis_dataset(DATA_DIR, 'Roman Urdu DataSet.csv', ['comment', 'sentiment', 'nan'])
    print(df.head(3))
