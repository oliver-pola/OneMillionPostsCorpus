# OneMillionPostsCorpus
# Deep Learning classification on a data set of german online discussions
# Student Project by: Jens Becker, Julius Plehn, Oliver Pola
#
# Building models with tf.keras

import numpy as np

from keras_preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences

import embedding


embedding_model = embedding.get_model()

vocab_size = embedding.vocab_size(embedding_model)
embedding_dim = embedding.embedding_dim(embedding_model)
# padded_length = 161 # holds the logest post from corpus, see wordcount.py
padded_length = 80 # probably enough to get the idea of a post

# corpus has CRLF and not LF only
preprocessing_filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r'


def preprocess(posts):
    encoded = [one_hot(post, vocab_size, filters=preprocessing_filters) for post in posts.tolist()]
    padded = pad_sequences(encoded, maxlen=padded_length, padding='post')
    return padded


def classifier():
    """ Classify a single catgegory
    """
    import tensorflow as tf # this shows some cuda message

    lstm_out = 128

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, 8, input_length=padded_length))
    model.add(tf.keras.layers.SpatialDropout1D(0.4))
    model.add(tf.keras.layers.LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def multi():
    """ Classify all catgegories at once
    """
    import tensorflow as tf # this shows some cuda message

    return None


if __name__ == '__main__':
	# Test
    model = classifier()
