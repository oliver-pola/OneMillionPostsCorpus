# OneMillionPostsCorpus
# Deep Learning classification on a data set of german online discussions
# Student Project by: Jens ?, Julius Plehn, Oliver Pola
#
# Access corpus data

import os
import wget
import tarfile
import sqlite3


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


if __name__ == '__main__':
	# Test
    all_tables = ['Articles', 'Posts', 'Newspaper_Staff', 'Annotations', 'Annotations_consolidated', 'CrossValSplit', 'Categories']
    conn = get_conn()
    print('Testing DB connection...')
    cur = conn.cursor()
    for table in all_tables:
        cur.execute(f'SELECT COUNT(*) FROM {table}')
        rows = cur.fetchall()
        for row in rows:
            print(f'{table} count = {row[0]}')
