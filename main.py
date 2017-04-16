from sklearn import linear_model
from collections import OrderedDict
from datasets import *
from linear.predictor import LinearRegression
from linear.classifier import LogisticRegression
from linear.newtons_classifier import NewtonsClassifier
from test_tools.regression_tester import RegressionTester
from test_tools.data_cache import DictionaryCache


_mask = 'A'
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
