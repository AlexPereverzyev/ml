import math
from linear.predictor import LinearRegression


class LogisticRegression(LinearRegression):
    """Basic classifier based on ordinary linear regression and sigmoid
    function"""

    def predict(self, data):
        """Evaluates target function using pre-trained parameters
        and provided sample"""
        result = super().predict(data)
        result = [1. / (1 + math.exp(-r)) for r in result]
        return result
