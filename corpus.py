# OneMillionPostsCorpus
# Deep Learning classification on a data set of german online discussions
# Student Project by: Jens ?, Julius Plehn, Oliver Pola
#
# Access corpus data

import os
import shutil
import wget
import tarfile
import sqlite3
import numpy as np


def get_conn():
    url = 'https://github.com/OFAI/million-post-corpus/releases/download/v1.0.0/million_post_corpus.tar.bz2'
    db = 'data/million_post_corpus/corpus.sqlite3'
    zip = os.path.split(url)[-1]
    if not os.path.exists(db):
        old_db = 'data/corpus.sqlite3'
        if os.path.exists(old_db):
            print('Rearrange data dir...')
            os.rename('data', 'million_post_corpus')
        else:
            if not os.path.exists(zip):
                print('Downloading corpus file...')
                wget.download(url, zip)
                print()
            print('Extract corpus DB...')
            with tarfile.open(zip) as tar:
                tar.extractall()
        os.makedirs('data', exist_ok=True)
        shutil.move('million_post_corpus', 'data')
        if os.path.exists(zip):
            os.remove(zip)
    conn = sqlite3.connect(db)
    return conn


def get_training(conn):
    cur = conn.cursor()
    x = [] # posts
    y = [] # categories
    # Select all posts that have any annotation, treating absent values as 0
    # sql =   ''' SELECT ID_Post, Headline, Body FROM Posts
    #             WHERE EXISTS
    #             (SELECT 1 FROM Annotations_consolidated
    #              WHERE Posts.ID_Post = Annotations_consolidated.ID_Post)
    #         '''
    # Select all posts that have annotations for each category
    sql =   ''' SELECT ID_Post, Headline, Body FROM Posts
                WHERE EXISTS
                (SELECT 1 FROM Annotations_consolidated
                 WHERE Posts.ID_Post = Annotations_consolidated.ID_Post
                 AND Category = 'SentimentNegative')
                AND EXISTS
                (SELECT 1 FROM Annotations_consolidated
                 WHERE Posts.ID_Post = Annotations_consolidated.ID_Post
                 AND Category = 'SentimentNeutral')
                AND EXISTS
                (SELECT 1 FROM Annotations_consolidated
                 WHERE Posts.ID_Post = Annotations_consolidated.ID_Post
                 AND Category = 'SentimentPositive')
                AND EXISTS
                (SELECT 1 FROM Annotations_consolidated
                 WHERE Posts.ID_Post = Annotations_consolidated.ID_Post
                 AND Category = 'OffTopic')
                AND EXISTS
                (SELECT 1 FROM Annotations_consolidated
                 WHERE Posts.ID_Post = Annotations_consolidated.ID_Post
                 AND Category = 'Inappropriate')
                AND EXISTS
                (SELECT 1 FROM Annotations_consolidated
                 WHERE Posts.ID_Post = Annotations_consolidated.ID_Post
                 AND Category = 'Discriminating')
                AND EXISTS
                (SELECT 1 FROM Annotations_consolidated
                 WHERE Posts.ID_Post = Annotations_consolidated.ID_Post
                 AND Category = 'PossiblyFeedback')
                AND EXISTS
                (SELECT 1 FROM Annotations_consolidated
                 WHERE Posts.ID_Post = Annotations_consolidated.ID_Post
                 AND Category = 'PersonalStories')
                AND EXISTS
                (SELECT 1 FROM Annotations_consolidated
                 WHERE Posts.ID_Post = Annotations_consolidated.ID_Post
                 AND Category = 'ArgumentsUsed')
            '''
    cur.execute(sql)
    for row in cur.fetchall():
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
    ## Ord = 6, Name = Discriminating
    ## Ord = 7, Name = PossiblyFeedback
    ## Ord = 8, Name = PersonalStories
    ## Ord = 9, Name = ArgumentsUsed
    cur = conn.cursor()
    categories = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    sql =   ''' SELECT Ord, Value FROM Annotations_consolidated
                JOIN Categories
                ON Annotations_consolidated.Category = Categories.Name
                WHERE ID_Post = ?
                ORDER BY Ord
            '''
    param = (id_post, )
    cur.execute(sql, param)
    for row in cur.fetchall():
        categories[row[0] - 1] = row[1]
    return categories


if __name__ == '__main__':
	# Test
    conn = get_conn()
    cur = conn.cursor()
    all_tables = ['Articles', 'Posts', 'Newspaper_Staff', 'Annotations', 'Annotations_consolidated', 'CrossValSplit', 'Categories']
    print('Test: Count tables...')
    for table in all_tables:
        cur.execute(f'SELECT COUNT(*) FROM {table}')
        for row in cur.fetchall():
            print(f'{table} count = {row[0]}')
    print()
    print('Test: List Categories...')
    cur.execute(f'SELECT Name, Ord FROM Categories')
    for row in cur.fetchall():
        print(f'Ord = {row[1]}, Name = {row[0]}')
    print()
    print('Test: Training data...')
    x, y = get_training(conn)
    print(f'x[0] = {x[0]}')
    print(f'y[0] = {y[0]}')
    print(f'Training count = {x.shape[0]}')
