# OneMillionPostsCorpus
# Deep Learning classification on a data set of german online discussions
# Student Project by: Jens Becker, Julius Plehn, Oliver Pola
#
# Word embedding

import os
import shutil
import wget
import tarfile
import numpy as np
import gensim

from gensim.scripts.glove2word2vec import glove2word2vec

import corpus # just to ensure data dir is rearranged


def get_model():
    """ Word2Vec Embedding German
        Refrence: https://deepset.ai/german-word-embeddings
    """
    # https://deepset.ai/german-word-embeddings
    # vocab_url = 'https://int-emb-word2vec-de-wiki.s3.eu-central-1.amazonaws.com/vocab.txt'
    # vectors_url = 'https://int-emb-word2vec-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt'
    # vocab = 'data/word2vec/vocab.txt'
    # vectors = 'data/word2vec/vectors.txt'

    # Download GloVe version and convert, since the Word2Vec download seems to be malformatted
    vectors_url = 'https://int-emb-glove-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt'
    vectors_glove = 'data/word2vec/vectors_glove.txt'
    vectors_txt = 'data/word2vec/vectors.txt'
    vectors = 'data/word2vec/vectors.bin'

    # https://github.com/devmount/GermanWordEmbeddings
    # vectors_url = 'http://cloud.devmount.de/d2bc5672c523b086/german.model'
    # vectors = 'data/word2vec/german.model'

    corpus.get_conn() # just to ensure data dir is rearranged
    os.makedirs('data/word2vec', exist_ok=True)
    # if not os.path.exists(vocab):
    #     print('Downloading Word2Vec vocabulary...')
    #     wget.download(vocab_url, vocab)
    #     print()
    # if not os.path.exists(vectors):
    #     print('Downloading Word2Vec vectors...')
    #     wget.download(vectors_url, vectors)
    #     print()

    # Use GloVe version an convert to Word2Vec
    if not os.path.exists(vectors):
        if not os.path.exists(vectors_txt):
            if not os.path.exists(vectors_glove):
                print('Downloading GloVe vectors...')
                wget.download(vectors_url, vectors_glove)
                print()
            print('Converting GloVe to Word2Vec...')
            glove2word2vec(vectors_glove, vectors_txt)
            os.remove(vectors_glove)
        print('Convert loading text...')
        model = gensim.models.KeyedVectors.load_word2vec_format(vectors_txt, binary=False)
        print('Convert writing binary...')
        model.save_word2vec_format(vectors, binary=True)
        os.remove(vectors_txt)
        del model

    print('Load Word2Vec embedding...')
    model = gensim.models.KeyedVectors.load_word2vec_format(vectors, binary=True)
    print(f'vocabulary size = {vocab_size(model)}')
    print(f'embedding dim = {embedding_dim(model)}')
    return model


def preprocess(list_of_words):
    return [w.lower() for w in list_of_words]


def vectors(model, list_of_words):
    filtered = filter(lambda w: w in model.vocab, preprocess(list_of_words))
    return [model[w] for w in filtered]


def indices(model, list_of_words):
    filtered = filter(lambda w: w in model.vocab, preprocess(list_of_words))
    return [model.vocab[w].index for w in filtered]


def vocab_size(model):
    return len(model.vocab)


def embedding_dim(model):
    return len(model[next(iter(model.vocab.keys()))])


def matrix(model):
    print('Build embedding matrix...')
    # index 0 does not represent a word
    embedding_matrix = np.zeros((vocab_size(model) + 1, embedding_dim(model)))
    for word, item in model.vocab.items():
        try:
            embedding_matrix[item.index] = model[word]
        except:
            print(i, word)
    print(f'embedding matrix shape = {embedding_matrix.shape}')
    return embedding_matrix


if __name__ == '__main__':
	# Test
    model = get_model()
    words = ['Dies', 'ist', 'ein', 'Test']
    # words = ['Dies', 'Test'] # not in vocabulary: ist, ein
    print(f'words = {words}')
    indi = indices(model, words)
    print(f'indices = {indi}')
    embedded = np.array(vectors(model, words))
    print(f'embedded.shape = {embedded.shape}')
