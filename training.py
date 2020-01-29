# OneMillionPostsCorpus
# Deep Learning classification on a data set of german online discussions
# Student Project by: Jens Becker, Julius Plehn, Oliver Pola
#
# Train the models

import sys
import numpy as np
import matplotlib.pyplot as plt

import corpus


def single_category(category):
    """ Trains a model for given single category
    """
    import models

    epochs = 50

    with corpus.get_conn() as conn:
        posts, label_vectors = corpus.get_training(conn)

    preprocessed = np.array(models.preprocess(posts))
    del posts
    print(f'preprocessed.shape = {preprocessed.shape}')

    category_index = corpus.categories[category]
    labels = np.array(label_vectors[:,category_index])
    del label_vectors
    print(f'labels.shape = {labels.shape}')

    model = models.classifier()
    history = model.fit(preprocessed, labels, epochs=epochs, verbose=2, validation_split=0.1)

    # plot history, https://keras.io/visualization
    plt.subplot(121)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.subplot(122)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


def all_categories():
    """ Trains a model for all categories at once
    """
    import models

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
