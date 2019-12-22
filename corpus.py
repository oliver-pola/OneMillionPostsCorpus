# OneMillionPostsCorpus
# Deep Learning classification on a data set of german online discussions
# Student Project by: Jens ?, Julius Plehn, Oliver Pola
#
# Access corpus data

import os
import wget
import tarfile
import sqlite3
import numpy as np


def get_conn():
    url = 'https://github.com/OFAI/million-post-corpus/releases/download/v1.0.0/million_post_corpus.tar.bz2'
    db = 'data/corpus.sqlite3'
    zip = os.path.split(url)[-1]
    if not os.path.exists(db):
        if not os.path.exists(zip):
            print('Downloading corpus file...')
            wget.download(url, zip)
            print()
        print('Extract corpus DB...')
        with tarfile.open(zip) as tar:
            tar.extractall()
        os.rename('million_post_corpus', 'data')
        os.remove(zip)
    conn = sqlite3.connect(db)
    return conn


def get_training(conn):
    cur = conn.cursor()
    x = [] # posts
    y = [] # categories
    sql =   f'''SELECT ID_Post, Headline, Body FROM Posts
                WHERE EXISTS
                (SELECT 1 FROM Annotations_consolidated
                 WHERE Posts.ID_Post = Annotations_consolidated.ID_Post)
            '''
    cur.execute(sql)
    rows = cur.fetchall()
    for row in rows:
        id_post = row[0]
        # merge Headline, Body to one text entry
        x.append(merge_text(row[1], row[2]))
        # categories for one post will be a vector, so y will be a matrix
        y.append(get_categorylist(conn, id_post))
    return np.array(x), np.array(y)


def merge_text(a, b, separator='\r\n'):
    if a == None or a == '':
        return b
    elif b == 'None' or b == '':
        return a
    else:
        return f'{a}{separator}{b}'


def get_categorylist(conn, id_post):
    ## Categories content:
    ## Ord = 1, Name = SentimentNegative
    ## Ord = 2, Name = SentimentNeutral
    ## Ord = 3, Name = SentimentPositive
    ## Ord = 4, Name = OffTopic
    ## Ord = 5, Name = Inappropriate
    ## Ord = 7, Name = PossiblyFeedback
    ## Ord = 8, Name = PersonalStories
    ## Ord = 9, Name = ArgumentsUsed
    cur = conn.cursor()
    categories = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    sql =   f'''SELECT Ord, Value FROM Annotations_consolidated
                JOIN Categories
                ON Annotations_consolidated.Category = Categories.Name
                WHERE ID_Post = {id_post}
                ORDER BY Ord
            '''
    cur.execute(sql)
    rows = cur.fetchall()
    for row in rows:
        categories[row[0] - 1] = row[1]
    return categories


if __name__ == '__main__':
	# Test
    conn = get_conn()
    all_tables = ['Articles', 'Posts', 'Newspaper_Staff', 'Annotations', 'Annotations_consolidated', 'CrossValSplit', 'Categories']
    print('Test: Count tables...')
    cur = conn.cursor()
    for table in all_tables:
        cur.execute(f'SELECT COUNT(*) FROM {table}')
        rows = cur.fetchall()
        for row in rows:
            print(f'{table} count = {row[0]}')
    print()
    print('Test: List Categories...')
    cur = conn.cursor()
    cur.execute(f'SELECT Name, Ord FROM Categories')
    rows = cur.fetchall()
    for row in rows:
        print(f'Ord = {row[1]}, Name = {row[0]}')
    print()
    print('Test: Training data...')
    x, y = get_training(conn)
    print(f'x[0] = {x[0]}')
    print(f'y[0] = {y[0]}')
    print(f'Training count = {x.shape[0]}')
