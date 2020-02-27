# OneMillionPostsCorpus
# Deep Learning classification on a data set of german online discussions
# Student Project by: Jens Becker, Julius Plehn, Oliver Pola
#
# Train the models

import sys
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import corpus

lstm_out_list = [32, 64, 96, 128, 160, 192]
dense_units_list = [8, 16, 32, 48, 64, 80, 96, 112, 128]
repeats = 3

def all_categories(epochs=50):
    """ Trains a model for all categories at once
    """
    import models
    import tensorflow as tf
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

    with corpus.get_conn() as conn:
        posts, label_vectors = corpus.get_training(conn)

    preprocessed = np.array(models.preprocess(posts))
    del posts
    print(f'preprocessed.shape = {preprocessed.shape}')

    labels = np.array(label_vectors)
    del label_vectors
    print(f'labels.shape = {labels.shape}')
    permutation = np.random.permutation(preprocessed.shape[0])
    preprocessed = preprocessed[permutation]
    labels = labels[permutation]

    val_split = 0.15
    val_count = int(np.round(preprocessed.shape[0] * val_split))
    print(f'val_count = {val_count}')
    print(f'train labels mean = {np.mean(labels[:-val_count], axis=0)}')
    print(f'val labels mean = {np.mean(labels[-val_count:], axis=0)}')

    class_occurances = np.count_nonzero(labels[:-val_count], axis=0)
    class_weights = class_occurances / np.sum(class_occurances)
    class_weights = dict(enumerate(class_weights))
    print(f'class_weights = {class_weights}')

    callbacks = [
        ReduceLROnPlateau(),
        EarlyStopping(patience=4)
    ]

    csv_file = f'output/All/training_meta.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as csv:
            csv.write('lstm_out, dense_units, F_1\n')

    for lstm_out in lstm_out_list:
        for dense_units in dense_units_list:
            for repeat_to_calc_means in range(repeats):
                model = models.multi(lstm_out=lstm_out, dense_units=dense_units, free_embedding_memory=False)

                history = model.fit(preprocessed, labels, callbacks=callbacks, epochs=epochs, verbose=2, validation_split=val_split, class_weight=class_weights, batch_size=16)

                val_labels = labels[-val_count:]
                print(f'val_labels.shape = {val_labels.shape}')
                val_predict = (model.predict(preprocessed[-val_count:]) > 0.5) * 1 # turn predictions into integers
                print(f'val_predict.shape = {val_predict.shape}')
                val_predict = val_predict.reshape(val_labels.shape)

                print('final validation results per category:')
                category = 'ArgumentsUsed'
                category_index = corpus.categories[category]

                cat_labels = val_labels[:,category_index]
                cat_predict = val_predict[:,category_index]
                eq = cat_labels == cat_predict
                neq = cat_labels != cat_predict

                tp = np.sum(eq[cat_predict == 1], axis=0)
                tn = np.sum(eq[cat_predict == 0], axis=0)
                fp = np.sum(neq[cat_predict == 1], axis=0)
                fn = np.sum(neq[cat_predict == 0], axis=0)
                accuracy = (tp + tn) / val_labels.shape[0]
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2.0 * precision * recall / (precision + recall)

                print(category)
                print(f'  true pos = {tp}')
                print(f'  true neg = {tn}')
                print(f'  false pos = {fp}')
                print(f'  false neg = {fn}')
                print(f'  accuracy = {accuracy:.4f}')
                print(f'  precision = {precision:.4f}')
                print(f'  recall = {recall:.4f}')
                print(f'  F_1 = {f1:.4f}')

                with open(csv_file, 'a') as csv:
                    csv.write(f'{lstm_out}, {dense_units}, {f1:.4f}\n')
                tf.keras.backend.clear_session()


def graph():
    csv_file = f'output/All/training_meta.csv'
    if not os.path.exists(csv_file):
        print('No data to build graph')
        return
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    plt.figure('training meta', figsize=(6, 4))
    plt.title('Evaluation for category = ''ArgumentsUsed''')
    for lstm_out in [64, 96, 128, 160]: # don't overload graph with lstm_out_list:
        graph_x = []
        graph_y = []
        graph_e = []
        lstm_data = data[data[:,0] == lstm_out]
        for dense_units in dense_units_list:
            dense_data = lstm_data[lstm_data[:,1] == dense_units]
            graph_x.append(dense_units)
            graph_y.append(np.mean(dense_data[:,2]))
            graph_e.append(np.std(dense_data[:,2]))
        plt.plot(graph_x, graph_y, label=f'{lstm_out} LSTM units')
        # plt.errorbar(graph_x, graph_y, yerr=graph_e, label=f'{lstm_out} LSTM units')
    plt.xlabel('dense units')
    plt.ylabel('$F_1$')
    plt.legend()
    plt.savefig(f'output/All/training_meta.png')
    plt.show()


def graph2():
    csv_file = f'output/All/training_meta.csv'
    if not os.path.exists(csv_file):
        print('No data to build graph')
        return
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    plt.figure('training meta 2', figsize=(6, 4))
    plt.title('Evaluation for category = ''ArgumentsUsed''')
    for dense_units in [32, 48, 64, 80]: # don't overload graph with dense_units_list:
        graph_x = []
        graph_y = []
        graph_e = []
        dense_data = data[data[:,1] == dense_units]
        for lstm_out in lstm_out_list:
            lstm_data = dense_data[dense_data[:,0] == lstm_out]
            graph_x.append(lstm_out)
            graph_y.append(np.mean(lstm_data[:,2]))
            graph_e.append(np.std(lstm_data[:,2]))
        plt.plot(graph_x, graph_y, label=f'{dense_units} dense units')
        # plt.errorbar(graph_x, graph_y, yerr=graph_e, label=f'{lstm_out} LSTM units')
    plt.xlabel('LSTM units')
    plt.ylabel('$F_1$')
    plt.legend()
    plt.savefig(f'output/All/training_meta2.png')
    plt.show()


class Logger(object):
    # see https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file
    # and https://stackoverflow.com/questions/20525587/python-logging-in-multiprocessing-attributeerror-logger-object-has-no-attrib
    def __init__(self, category):
        self.terminal = sys.stdout
        self.log = open(f'output/{category}/training_meta.log', 'w')

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
    elif sys.argv[1] == 'Graph':
        graph()
    elif sys.argv[1] == 'Graph2':
        graph2()
    elif sys.argv[1] not in ['All'] + list(corpus.categories):
        usage()
    else:
        category = sys.argv[1]
        init(category)
        if len(sys.argv) == 3:
            epochs = int(sys.argv[2])
        else:
            epochs = 50
        if category == 'All':
            all_categories(epochs)
        else:
            print('Meta training for single category not implemented')


def usage():
    categories = ' | '.join(list(corpus.categories))
    print(f'Usage: python {sys.argv[0]} < All | Category > [ Epochs = 50 ]')
    print(f'Category: {categories}')


if __name__ == '__main__':
	run()
