# OneMillionPostsCorpus
# Deep Learning classification on a data set of german online discussions
# Student Project by: Jens Becker, Julius Plehn, Oliver Pola
#
# LSTM model for category SentimentNegative only

import sqlite3
from sqlite3 import Error

import multiprocessing
import os
import tensorflow as tf

from keras_preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences

import numpy as np

import corpus




def select_all_labled_posts(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute('''
        SELECT ID_Post, COALESCE(Headline, '') || ' ' || COALESCE(Body, '')
        FROM Posts
        WHERE ID_Post IN (
            SELECT DISTINCT ID_Post
            FROM Annotations
            WHERE Category = 'SentimentNegative')
        ''')

    rows = cur.fetchall()

    posts = 'test'

    for row in rows:
        print(row)

    return rows


def select_all_negative_sentiments(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute('''
        SELECT Value FROM Annotations_consolidated WHERE Category='SentimentNegative';
        ''')

    rows = cur.fetchall()


    for row in rows:
        print(row)

    return rows


def longest(l):
    if not isinstance(l, list): return(0)
    return(max([len(l),] + [len(subl) for subl in l if isinstance(subl, list)] +
        [longest(subl) for subl in l]))



def main():
    conn = corpus.get_conn()

    with conn:
        # posts = select_all_labled_posts(conn)
        # neg_posts = select_all_negative_sentiments(conn)
        x, y = corpus.get_training(conn)

    # docs = []
    # labels = []

    # neg_sent = []
    #
    # for n in neg_posts:
    #     neg_sent.append(n)
    #
    # for d in posts:
    #     docs.append(d[1])
    #
    # labels = np.asarray(neg_sent)
    labels = y[:,0] # only SentimentNegative
    #np.random.seed(1)
    #labels = np.random.randint(0, 2, 2500)
    # print(labels)

    vocab_size = 5000
    # encoded_docs = [one_hot(d, vocab_size) for d in docs]
    encoded_docs = [one_hot(d, vocab_size) for d in x.tolist()]

    max_length = 100
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs)

    lstm_out=128

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, 8, input_length=max_length))
    model.add(tf.keras.layers.SpatialDropout1D(0.4))
    model.add(tf.keras.layers.LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    model.fit(padded_docs, labels, epochs = 50, verbose = 2, validation_split=0.1)

if __name__ == '__main__':
    main()
