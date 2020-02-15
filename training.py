# OneMillionPostsCorpus
# Deep Learning classification on a data set of german online discussions
# Student Project by: Jens Becker, Julius Plehn, Oliver Pola
#
# Train the models

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import corpus


def single_category(category, epochs=100):
    """ Trains a model for given single category
    """
    import models

    with corpus.get_conn() as conn:
        posts, label_vectors = corpus.get_training(conn)

    preprocessed = np.array(models.preprocess(posts))
    del posts
    print(f'preprocessed.shape = {preprocessed.shape}')

    category_index = corpus.categories[category]
    labels = np.array(label_vectors[:,category_index])
    del label_vectors
    print(f'labels.shape = {labels.shape}')

    # shuffle data and labels with same permutation
    # because model.fit() will take validation data from the end and shuffle afterwards
    permutation = np.random.permutation(preprocessed.shape[0])
    preprocessed = preprocessed[permutation]
    labels = labels[permutation]

    val_split = 0.1
    val_count = int(np.round(preprocessed.shape[0] * val_split))
    print(f'val_count = {val_count}')
    print(f'train labels mean = {np.mean(labels[:-val_count], axis=0)}')
    print(f'val labels mean = {np.mean(labels[-val_count:], axis=0)}')

    model = models.classifier()
    history = model.fit(preprocessed, labels, epochs=epochs, verbose=2, validation_split=val_split)
    model.save(f'output/{category}/model.h5')
    # print(history.history)
    plot_hist(history, category)


def all_categories(epochs=100):
    """ Trains a model for all categories at once
    """
    import models

    with corpus.get_conn() as conn:
        posts, label_vectors = corpus.get_training(conn)

    preprocessed = models.preprocess(posts)

    model = models.multi()
    # model.fit(preprocessed, label_vectors, epochs=epochs, verbose=2, validation_split=0.1)
    # model.save('output/All/model.h5')
    # plot_hist(history, 'All')


def arrange_cols_rows(num, aspect=1):
    cols = int(np.floor(np.sqrt(aspect * num)))
    rows = (num + cols - 1) // cols
    return cols, rows


def plot_hist(history, category):
    # plot history, https://keras.io/visualization
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    cols, rows = arrange_cols_rows(len(metrics))
    plt.figure('training', figsize=(10, 10 * cols / rows))
    plt.suptitle(category)
    for i, metric in enumerate(metrics):
        plt.subplot(cols, rows, i + 1)
        plt.plot(history.history[metric], label='training')
        plt.plot(history.history['val_' + metric], label='validation')
        plt.title(metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend()
    plt.savefig(f'output/{category}/training.png')
    plt.show()


class Logger(object):
    # see https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file
    # and https://stackoverflow.com/questions/20525587/python-logging-in-multiprocessing-attributeerror-logger-object-has-no-attrib
    def __init__(self, category):
        self.terminal = sys.stdout
        self.log = open(f'output/{category}/training.log', 'w')

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def init(category):
    os.makedirs(f'output/{category}', exist_ok=True)
    sys.stdout = Logger(category)


def run():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        usage()
    elif len(sys.argv) == 3 and not sys.argv[2].isdigit():
        usage()
    elif sys.argv[1] not in ['All'] + list(corpus.categories):
        usage()
    else:
        category = sys.argv[1]
        init(category)
        if len(sys.argv) == 3:
            epochs = int(sys.argv[2])
        else:
            epochs = 100
        if category == 'All':
            all_categories(epochs)
        else:
            single_category(category, epochs)


def usage():
    categories = ' | '.join(list(corpus.categories))
    print(f'Usage: python {sys.argv[0]} < All | Category > [ Epochs = 100 ]')
    print(f'Category: {categories}')


if __name__ == '__main__':
	run()
