# OneMillionPostsCorpus
# Deep Learning classification on a data set of german online discussions
# Student Project by: Jens Becker, Julius Plehn, Oliver Pola
#
# Statistics on word count of posts

import numpy as np

from keras_preprocessing.text import text_to_word_sequence

import corpus


def tokenize(posts):
    posts = list(posts) # in case it's numpy array
    return [text_to_word_sequence(p) for p in posts]


if __name__ == '__main__':
	# Test
    conn = corpus.get_conn()
    x, y = corpus.get_training(conn)
    print('Test: CRLF vs LF...')
    text = 'Test mit einer Zeile, \r\neiner zweiten und auch CRLF am Ende\r\n'
    print(f'text = {repr(text)}')
    # default filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    print(f'default filtered = {text_to_word_sequence(text)}')
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r'
    print(f'custom filtered = {text_to_word_sequence(text, filters=filters)}')
    print(f'real post = {repr(x[49])}')
    print(f'default filtered = {text_to_word_sequence(x[49])}')
    print(f'custom filtered = {text_to_word_sequence(x[49], filters=filters)}')
    print()
    print('Test: Word count...')
    tokenized = tokenize(x)
    count = []
    # badcount = 200
    # badindex = 0
    for i, tokens in enumerate(tokenized):
        l = len(tokens)
        count.append(l)
        if l == 0:
            print(f'post resulting in 0 words = {repr(x[i])}')
        # if '\r' in x[i] and l < badcount and l > 5:
        #     badcount = l
        #     badindex = i
    count = np.array(count)
    # print(f'CRLF example at index {badindex}')
    print(f'min word count = {np.min(count)}')
    print(f'max word count = {np.max(count)}')
    print(f'avg word count = {np.mean(count):.2f}')
