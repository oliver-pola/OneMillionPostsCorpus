# OneMillionPostsCorpus
# Deep Learning classification on a data set of german online discussions
# Student Project by: Jens ?, Julius Plehn, Oliver Pola
#
# Word embedding

import os
import shutil
import wget
import tarfile
import numpy as np


def get_embedding():
    """ Word2Vec Embedding German
        Refrence: https://deepset.ai/german-word-embeddings
    """
    vocab_url = 'https://int-emb-word2vec-de-wiki.s3.eu-central-1.amazonaws.com/vocab.txt'
    vectors_url = 'https://int-emb-word2vec-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt'
    vocab = 'data/word2vec/vocab.txt'
    vectors = 'data/word2vec/vectors.txt'
    os.makedirs('data/word2vec')
    if not os.path.exists(vocab):
        print('Downloading Word2Vec vocabulary...')
        wget.download(vocab_url, vocab)
        print()
    if not os.path.exists(vectors):
        print('Downloading Word2Vec vectors...')
        wget.download(vectors_url, vectors)
        print()
    return True


if __name__ == '__main__':
	# Test
    get_embedding()
