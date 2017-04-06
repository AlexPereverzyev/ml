from predictor import LinearRegression
from classifier import LogisticRegression
from training_sets import *


if __name__ == "__main__":
    test_regression('Linear Regression', 1, 100, 0.00001, 1.5,
                    TrainingSet1, ValidationSet1, 0, LinearRegression)

    test_regression('Logistic Regression', 2, 1000, 0.0005, .1,
                    TrainingSet2, ValidationSet2, 0, LogisticRegression)

    # test_regression('Response Time Logistics', 11, 1000, 0.0000001, .1,
    #                 TrainingSet3, ValidationSet3, 100, LogisticRegression)
