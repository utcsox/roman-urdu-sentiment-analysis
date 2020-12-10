"""modular to vectorize data
Converts the cleaned text into numeric representation
"""

import emoji
import functools
import pandas as pd
import nltk
import operator
import re
import string


def tokenizer(doc: pd.Series) -> str:
    """
    Tokenize a single document
    :param doc:
    :return:
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
