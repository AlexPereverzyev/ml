from predictor import LinearRegression
from classifier import LogisticRegression
from newtons_classifier import NewtonsClassifier
from training_sets import *


if __name__ == "__main__":
    test_regression('Linear Regression', 1, 100, 0.00001, 1.5,
                    TrainingSet1, ValidationSet1, 0, LinearRegression)

    test_regression('Logistic Regression 1', 2, 100, 0.0005, .1,
                    TrainingSet2, ValidationSet2, 0, LogisticRegression)

    test_regression('Logistic Regression 2', 2, 100, 0.5, .1,
                    TrainingSet4, ValidationSet4, 0, LogisticRegression)

    test_regression('Logistic Regression 2 (N-R)', 2, 5, 1, .1,
                    TrainingSet4, ValidationSet4, 0, NewtonsClassifier)

    # test_regression('Failure Probability', 11, 1000, 0.0000001, .1,
    #                 TrainingSet3, ValidationSet3, 0, LogisticRegression)
