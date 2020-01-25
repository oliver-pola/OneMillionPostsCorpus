# OneMillionPostsCorpus
# Deep Learning classification on a data set of german online discussions
# Student Project by: Jens Becker, Julius Plehn, Oliver Pola
#
# Train the models

import sys
import numpy as np

import corpus
import models


def single_category(category):
    """ Trains a model for given single category
    """
    epochs = 50

    with corpus.get_conn() as conn:
        posts, label_vectors = corpus.get_training(conn)

    preprocessed = models.preprocess(posts)

    category_index = corpus.categories[category]
    labels = label_vectors[:,category_index]

    model = models.classifier()
    model.fit(preprocessed, labels, epochs=epochs, verbose=2, validation_split=0.1)


def all_categories():
    """ Trains a model for all categories at once
    """
    epochs = 50

    with corpus.get_conn() as conn:
        posts, label_vectors = corpus.get_training(conn)

    preprocessed = models.preprocess(posts)
    
    model = models.multi()
    # model.fit(preprocessed, label_vectors, epochs=epochs, verbose=2, validation_split=0.1)


def run():
    if len(sys.argv) != 2:
        usage()
    elif sys.argv[1] == 'All':
        all_categories()
    elif sys.argv[1] not in list(corpus.categories):
        usage()
    else:
        single_category(sys.argv[1])


def usage():
    categories = ' | '.join(list(corpus.categories))
    print(f'Usage: python {sys.argv[0]} < All | Category >')
    print(f'Category: {categories}')


if __name__ == '__main__':
	run()