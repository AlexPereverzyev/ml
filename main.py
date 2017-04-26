from collections import OrderedDict
from datasets import *
from sklearn import linear_model
from sklearn import naive_bayes
from linear.predictor import LinearRegression
from linear.classifier import LogisticRegression
from linear.newtons_classifier import NewtonsClassifier
from generative.naive_bayes import NaiveBayesClassifier
from generative.gaussian import GaussianClassifier
from test_tools.regression_tester import RegressionTester
from test_tools.generative_tester import GenerativeTester
from test_tools.data_cache import DictionaryCache

# from data_tools.data_plotter import DataPlotter
# DataPlotter.plot_all(
#     """
#         data/CLR-Locks and Threads-Current Physical Threads.csv|@date,value;
#         data/IIS-CPU.csv|@date,value;
#         data/Calls Per Minute.csv|@date,value;
#         data/Errors Per Minute.csv|@date,value;
#         data/Average Response Time.csv|@date,value;
#     """)

_mask = 'G'
_rs = OrderedDict([
    ('Naive Bayes (Gaussian)',
        (GaussianClassifier([0., 1.], 2),
         TrainingSet5, ValidationSet5, .1, 'G', GenerativeTester)),

    ('Naive Bayes (Gaussian-SKL)',
        (naive_bayes.GaussianNB(),
         TrainingSet5, ValidationSet5, .1, 'G', GenerativeTester)),

    ('Naive Bayes (multinomial)',
        (NaiveBayesClassifier([0., 1.], 12),
         TrainingSet6, ValidationSet6, .1, 'G', GenerativeTester)),

    ('Naive Bayes (multinomial-SKL)',
        (naive_bayes.MultinomialNB(),
         TrainingSet6, ValidationSet6, .1, 'G', GenerativeTester)),

    ('Linear Regression',
        (LinearRegression(1, 100, 0.00001),
         TrainingSet1, ValidationSet1, 1, 'S', RegressionTester)),

    ('Linear Regression (SKL)',
        (linear_model.LinearRegression(),
         TrainingSet1, ValidationSet1, 3, 'S', RegressionTester)),

    ('Logistic Regression 1',
        (LogisticRegression(2, 100, 0.0005),
         TrainingSet2, ValidationSet2, .1, 'S', RegressionTester)),

    ('Logistic Regression 1 (SKL)',
        (linear_model.LogisticRegression(solver='liblinear'),
         TrainingSet2, ValidationSet2, .1, 'S', RegressionTester)),

    ('Logistic Regression 2',
        (LogisticRegression(2, 100, 0.5),
         TrainingSet4, ValidationSet4, .1, 'A', RegressionTester)),

    ('Logistic Regression 2 (SKL)',
        (linear_model.LogisticRegression(solver='liblinear'),
         TrainingSet4, ValidationSet4, .1, 'A', RegressionTester)),

    ('Logistic Regression 2 (NR)',
        (NewtonsClassifier(2, 5, 1),
         TrainingSet4, ValidationSet4, .1, 'A', RegressionTester)),

    ('Logistic Regression 2 (NR-SKL)',
        (linear_model.LogisticRegression(solver='newton-cg'),
         TrainingSet4, ValidationSet4, .1, 'A', RegressionTester)),

    ('Slowness Probability',
        (LogisticRegression(11, 1000, 0.0000001),
         TrainingSet3, ValidationSet3, .1, 'L', RegressionTester)),

    ('Slowness Probability (SKL)',
        (linear_model.LogisticRegression(solver='liblinear'),
         TrainingSet3, ValidationSet3, .1, 'L', RegressionTester)),

    ('Slowness Probability (NR-SKL)',
        (linear_model.LogisticRegression(solver='newton-cg'),
         TrainingSet3, ValidationSet3, .1, 'L', RegressionTester)),
])
_sc = DictionaryCache()

for t, ts, vs in ((tester(k, r, _sc, p), ts, vs)
                  for k, (r, ts, vs, p, flags, tester) in _rs.items()
                  if _mask == '*' or any(f in flags for f in _mask)):
    t.train(ts)
    t.printr()
    t.prints(ts)
    t.prints(vs)
    t.printy(vs, 10)
