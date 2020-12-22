"""modular to vectorize data
Converts the cleaned text into numeric representation
"""

from typing import Any, Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

MIN_DOCUMENT_FREQUENCY = 2
TOP_K = 30000
MAX_SEQUENCE_LENGTH = 300


def vectorize(train_texts: List[str], train_labels, test_texts: List[str]) -> Tuple[Any, Any]:
    """ Convert the document into word n-grams and vectorize it with tf-idf

    :param train_texts: of training texts
    :param train_labels: An array of labels from the training dataset
    :param test_texts: List of test texts
    :return: A tuple of vectorize training_text and vectorize test texts
    """

    kwargs = {
        'ngram_range': (1, 2),
        'dtype': 'int32',
        'analyzer': 'word',
        'min_df': MIN_DOCUMENT_FREQUENCY
    }
    # Use TfidfVectorizer to convert the raw documents to a matrix of TF-IDF features with:
    # either 1-gram or 2-gram, using 'word' to split, and minimum document/corpus frequency of 2
    # Limit the number of features to top 30K.

    vectorizer = TfidfVectorizer(**kwargs)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    selector = SelectKBest(f_classif, k=min(30000, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train)
    x_test = selector.transform(x_test)

    x_train = x_train.astype('float32').toarray()
    x_test = x_test.astype('float32').toarray()

    return x_train, x_test


def tf_keras_vectorize(train_texts: List[str], test_texts: List[str]) -> Tuple[Any, Any, Dict[str, int]]:
    """ Vectorize text with tf.keras.preprocessing class

    :param train_texts: of training texts
    :param test_texts: List of test texts
    :return: A tuple of vectorize training_text, vectorize test texts and a dictionary of word and index
    """
    # create vocabulary with training texts
    tokenizer = Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # vectorize the training/test texts
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_test = tokenizer.texts_to_sequences(test_texts)

    # Get max sequence length
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    x_train = pad_sequences(x_train, maxlen=max_length)
    x_test = pad_sequences(x_test, maxlen=max_length)
    return x_train, x_test, tokenizer.word_index
