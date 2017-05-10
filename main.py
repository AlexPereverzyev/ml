from collections import OrderedDict
from datasets import *
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn.neural_network import MLPClassifier
from linear.predictor import LinearRegression
from linear.classifier import LogisticRegression
from linear.newtons_classifier import NewtonsClassifier
from generative.naive_bayes import NaiveBayesClassifier
from generative.gaussian import GaussianClassifier
from descriminant.linear_classifier import LinearDescriminantClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from neural_net.mlp_classifier import MultiLayerPerceptronClassifier
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

_mask = 'SNG'
_rs = OrderedDict([
    ('Multi-Layer Perceptron Classifier',
        (MultiLayerPerceptronClassifier(2, layers=(3,), iterations=250),
         TrainingSet5, TestSet5, .2, 'N', RegressionTester)),

    ('Multi-Layer Perceptron Classifier (SKL)',
        (MLPClassifier(hidden_layer_sizes=(3,), random_state=1),
         TrainingSet5, TestSet5, .2, 'N', RegressionTester)),

    ('Linear Descriminant Classifier',
        (LinearDescriminantClassifier([0., 1.], 2),
         TrainingSet5, TestSet5, .1, 'D', GenerativeTester)),

    ('Linear Descriminant Classifier (SKL)',
        (LinearDiscriminantAnalysis(solver='eigen'),
         TrainingSet5, TestSet5, .1, 'D', GenerativeTester)),

    ('Naive Bayes (Gaussian)',
        (GaussianClassifier([0., 1.], 2),
         TrainingSet5, TestSet5, .1, 'G', GenerativeTester)),

    ('Naive Bayes (Gaussian-SKL)',
        (naive_bayes.GaussianNB(),
         TrainingSet5, TestSet5, .1, 'G', GenerativeTester)),

    ('Naive Bayes (multinomial)',
        (NaiveBayesClassifier([0., 1.], 12),
         TrainingSet6, TestSet6, .1, 'G', GenerativeTester)),

    ('Naive Bayes (multinomial-SKL)',
        (naive_bayes.MultinomialNB(),
         TrainingSet6, TestSet6, .1, 'G', GenerativeTester)),

    ('Linear Regression',
        (LinearRegression(1, 100, 0.00001),
         TrainingSet1, TestSet1, 1, 'S', RegressionTester)),

    ('Linear Regression (SKL)',
        (linear_model.LinearRegression(),
         TrainingSet1, TestSet1, 3, 'S', RegressionTester)),

    ('Logistic Regression 1',
        (LogisticRegression(2, 100, 0.0005),
         TrainingSet2, TestSet2, .1, 'S', RegressionTester)),

    ('Logistic Regression 1 (SKL)',
        (linear_model.LogisticRegression(solver='liblinear'),
         TrainingSet2, TestSet2, .1, 'S', RegressionTester)),

    ('Logistic Regression 2',
        (LogisticRegression(2, 100, 0.5),
         TrainingSet4, TestSet4, .1, 'A', RegressionTester)),

    ('Logistic Regression 2 (SKL)',
        (linear_model.LogisticRegression(solver='liblinear'),
         TrainingSet4, TestSet4, .1, 'A', RegressionTester)),

    ('Logistic Regression 2 (NR)',
        (NewtonsClassifier(2, 5, 1),
         TrainingSet4, TestSet4, .1, 'A', RegressionTester)),

    ('Logistic Regression 2 (NR-SKL)',
        (linear_model.LogisticRegression(solver='newton-cg'),
         TrainingSet4, TestSet4, .1, 'A', RegressionTester)),

    ('Slowness Probability',
        (LogisticRegression(11, 1000, 0.0000001),
         TrainingSet3, TestSet3, .1, 'L', RegressionTester)),

    ('Slowness Probability (SKL)',
        (linear_model.LogisticRegression(solver='liblinear'),
         TrainingSet3, TestSet3, .1, 'L', RegressionTester)),

    ('Slowness Probability (NR-SKL)',
        (linear_model.LogisticRegression(solver='newton-cg'),
         TrainingSet3, TestSet3, .1, 'L', RegressionTester)),
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
