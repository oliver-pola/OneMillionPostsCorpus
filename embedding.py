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

import corpus # just to ensure data dir is rearranged


def get_model():
    """ Word2Vec Embedding German
        Refrence: https://github.com/devmount/GermanWordEmbeddings
    """
    # https://deepset.ai/german-word-embeddings
    # vocab_url = 'https://int-emb-word2vec-de-wiki.s3.eu-central-1.amazonaws.com/vocab.txt'
    # vectors_url = 'https://int-emb-word2vec-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt'
    # vocab = 'data/word2vec/vocab.txt'
    # vectors = 'data/word2vec/vectors.txt'
    vectors_url = 'http://cloud.devmount.de/d2bc5672c523b086/german.model'
    vectors = 'data/word2vec/german.model'
    corpus.get_conn() # just to ensure data dir is rearranged
    os.makedirs('data/word2vec', exist_ok=True)
    # if not os.path.exists(vocab):
    #     print('Downloading Word2Vec vocabulary...')
    #     wget.download(vocab_url, vocab)
    #     print()
    if not os.path.exists(vectors):
        print('Downloading Word2Vec vectors...')
        wget.download(vectors_url, vectors)
        print()
    print('Load Word2Vec embedding...')
    model = gensim.models.KeyedVectors.load_word2vec_format(vectors, binary=True)
    print(f'vocabulary size = {len(model.vocab)}')
    return model


def vectors(model, list_of_words):
    filtered = filter(lambda w: w in model.vocab, list_of_words)
    return np.array([model[w] for w in filtered])


def indices(model, list_of_words):
    filtered = filter(lambda w: w in model.vocab, list_of_words)
    return np.array([model.vocab[w].index for w in filtered])


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
    embedded = vectors(model, words)
    print(f'embedded.shape = {embedded.shape}')
