# OneMillionPostsCorpus
Deep Learning classification on a data set of german online discussions

Student Project by: Jens Becker, Julius Plehn, Oliver Pola

In Seminar: [Deep Learning for Language and Speech Seminar 19/20](https://www.inf.uni-hamburg.de/en/inst/ab/lt/teaching/ma-lectures/dl-seminar1718.html), Language Technology Group, Uni Hamburg

Data Set: [One Million Posts Corpus](https://ofai.github.io/million-post-corpus/)

References:

Dietmar Schabus, Marcin Skowron, Martin Trapp
[One Million Posts: A Data Set of German Online Discussions](https://github.com/OFAI/million-post-corpus/raw/gh-pages/assets/SIGIR_2017_preprint.pdf)
Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)
pp. 1241-1244
Tokyo, Japan, August 2017

Dietmar Schabus and Marcin Skowron
[Academic-Industrial Perspective on the Development and Deployment of a Moderation System for a Newspaper Website](http://www.lrec-conf.org/proceedings/lrec2018/pdf/8885.pdf)
Proceedings of the 11th International Conference on Language Resources and Evaluation (LREC 2018)
pp. 1602-1605
Miyazaki, Japan, May 2018

---

## Setup
The corpus script will download, extract and mount the SQLite DB. Run something like this in the project folder:

```
pip install pipenv
pipenv install
pipenv shell
python corpus.py
```

The required Python modules are listed in the provided `Pipfile` and installed via the `pipenv install` command.

To test the DB connection content of all corpus tables is counted and should look like this:

```
Articles count = 12087
Posts count = 1011773
Newspaper_Staff count = 110
Annotations count = 58568
Annotations_consolidated count = 40567
CrossValSplit count = 40567
Categories count = 9
```
