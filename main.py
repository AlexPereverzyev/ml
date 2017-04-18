from sklearn import linear_model
from collections import OrderedDict
from datasets import *
from linear.predictor import LinearRegression
from linear.classifier import LogisticRegression
from linear.newtons_classifier import NewtonsClassifier
from test_tools.regression_tester import RegressionTester
from test_tools.data_cache import DictionaryCache

# todo: refactor
from data_tools.data_iter import DataIterator
from generative.gaussian import GaussianClassifier
from sklearn.naive_bayes import GaussianNB
with DataIterator(TrainingSet5) as feed:
    data = list(feed)
gc = GaussianClassifier([0, 1], 2)
gc.train(data)
print(gc)
gnb = GaussianNB()
gnb.fit([s[:-1] for s in data], [s[-1] for s in data])
with DataIterator(ValidationSet5) as feed:
    data = list(feed)
for t1, t2 in zip(gc.predict_proba(data),
                  gnb.predict_proba([s[:-1] for s in data])):
    print('MY : {0:.4f} {1:.4f}\nSKL: {2:.4f} {3:.4f}'
          .format(*t1, *t2))
print()

from data_tools.data_iter import DataIterator
from generative.naive_bayes import NaiveBayesClassifier
from sklearn.naive_bayes import MultinomialNB
with DataIterator(TrainingSet6) as feed:
    data = list(feed)
nbc = NaiveBayesClassifier([0, 1], 12)
nbc.train(data)
print(nbc)
mnb = MultinomialNB()
mnb.fit([s[:-1] for s in data], [s[-1] for s in data])
with DataIterator(ValidationSet6) as feed:
    data = list(feed)
for t1, t2 in zip(nbc.predict_proba(data),
                  mnb.predict_proba([s[:-1] for s in data])):
    print('MY : {0:.4f} {1:.4f}\nSKL: {2:.4f} {3:.4f}'
          .format(*t1, *t2))
print()

_mask = '?'
_rs = OrderedDict([
    ('Linear Regression',
        (LinearRegression(1, 100, 0.00001),
         TrainingSet1, ValidationSet1, 1, 'S')),

    ('Linear Regression (SKL)',
        (linear_model.LinearRegression(),
         TrainingSet1, ValidationSet1, 3, 'S')),

    ('Logistic Regression 1',
        (LogisticRegression(2, 100, 0.0005),
         TrainingSet2, ValidationSet2, .1, 'S')),

    ('Logistic Regression 1 (SKL)',
        (linear_model.LogisticRegression(solver='liblinear'),
         TrainingSet2, ValidationSet2, .1, 'S')),

    ('Logistic Regression 2',
        (LogisticRegression(2, 100, 0.5),
         TrainingSet4, ValidationSet4, .1, 'A')),

    ('Logistic Regression 2 (SKL)',
        (linear_model.LogisticRegression(solver='liblinear'),
         TrainingSet4, ValidationSet4, .1, 'A')),

    ('Logistic Regression 2 (NR)',
        (NewtonsClassifier(2, 5, 1),
         TrainingSet4, ValidationSet4, .1, 'A')),

    ('Logistic Regression 2 (NR-SKL)',
        (linear_model.LogisticRegression(solver='newton-cg'),
         TrainingSet4, ValidationSet4, .1, 'A')),

    ('Slowness Probability',
        (LogisticRegression(11, 1000, 0.0000001),
         TrainingSet3, ValidationSet3, .1, 'L')),

    ('Slowness Probability (SKL)',
        (linear_model.LogisticRegression(solver='liblinear'),
         TrainingSet3, ValidationSet3, .1, 'L')),

    ('Slowness Probability (NR-SKL)',
        (linear_model.LogisticRegression(solver='newton-cg'),
         TrainingSet3, ValidationSet3, .1, 'L')),
])
_sc = DictionaryCache()

for t, ts, vs in ((RegressionTester(k, r, _sc, p), ts, vs)
                  for k, (r, ts, vs, p, flags) in _rs.items()
                  if any(f in flags for f in _mask)):
    t.train(ts)
    t.printr()
    t.prints(ts)
    t.prints(vs)
    t.printy(vs, 0)
