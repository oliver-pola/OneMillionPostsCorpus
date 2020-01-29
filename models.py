# OneMillionPostsCorpus
# Deep Learning classification on a data set of german online discussions
# Student Project by: Jens Becker, Julius Plehn, Oliver Pola
#
# Building models with tf.keras

import numpy as np

from keras_preprocessing.text import text_to_word_sequence
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
    word_lists = [text_to_word_sequence(post, filters=preprocessing_filters) for post in posts.tolist()]
    embedded = [embedding.indices(embedding_model, word_list) for word_list in word_lists]
    del word_lists
    padded = pad_sequences(embedded, maxlen=padded_length, padding='post')
    del embedded
    return padded
    # vectors = [embedding.vectors_from_indices(embedding_model, sequence) for sequence in padded]
    # return vectors


def classifier():
    """ Classify a single catgegory
    """
    import tensorflow as tf # this shows some cuda message

    lstm_out = 128

    global embedding_model
    embedding_matrix = embedding.matrix(embedding_model)
    del embedding_model

    with tf.distribute.MirroredStrategy().scope():
        model = tf.keras.Sequential()
        # index 0 does not represent a word -> vocab_size + 1
        model.add(tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=padded_length, trainable=False,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix)))
        del embedding_matrix
        # model.add(tf.keras.Input(shape=(padded_length, embedding_dim)))
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
