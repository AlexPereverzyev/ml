import math
from predictor import LinearRegression


class LogisticRegression(LinearRegression):
    """Basic classifier based on ordinary linear regression and sigmoid
    function"""

    def predict(self, sample):
        """Evaluates target function using pre-trained parameters
        and provided sample"""
        result = super().predict(sample)
        result = 1. / (1 + math.exp(-result))
        return result
