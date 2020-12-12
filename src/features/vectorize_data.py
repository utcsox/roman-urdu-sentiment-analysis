"""modular to vectorize data
Converts the cleaned text into numeric representation
"""
import numpy as np

from typing import Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

MIN_DOCUMENT_FREQUENCY = 2
TOP_K = 30000

def vectorize(train_texts: List[str], train_labels, test_texts: List[str]) -> Tuple[Any, Any]:
    """ Convert the document into word n-grams and vectorize it

    :param train_texts: of training texts
    :param train_labels: An array of labels from the training dataset
    :param test_texts: List of test texts
    :return: A tuple of vectorize training_text and vectorize test texts
    """

    kwargs = {
        'ngram_range': (1, 2),
        'analyzer': 'word',
        'min_df': MIN_DOCUMENT_FREQUENCY
    }
    # Use TfidfVectorizer to convert the raw documents to a matrix of TF-IDF features with:
    # either 1-gram or 2-gram, using 'word' to split, and minimum document/corpus frequency of 2
    # Limit the number of features to top 30K.

    vectorizer = TfidfVectorizer(**kwargs)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    selector = SelectKBest(f_classif, k=min(30000, X_train.shape[1]))
    selector.fit(X_train, train_labels)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    return X_train, X_test
