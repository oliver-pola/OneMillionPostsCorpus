# OneMillionPostsCorpus
# Deep Learning classification on a data set of german online discussions
# Student Project by: Jens Becker, Julius Plehn, Oliver Pola
#
# Statistics about the three sentiment categories

import numpy as np

import corpus


if __name__ == '__main__':
    # Test
    conn = corpus.get_conn()
    x, y = corpus.get_training(conn)
    # only firt 3 gategories = SentimentX
    y = y[:,:3]
    # check if none, always one exclusive (hopefully) or multiple sentiments set
    sums = np.sum(y, axis=1)
    print(f'number of posts = {sums.shape[0]}')
    print(f'number without sentiment = {sums[sums == 0].shape[0]}')
    print(f'number with 1 sentiment = {sums[sums == 1].shape[0]}')
    print(f'number with >1 sentiments = {sums[sums > 1].shape[0]}')
