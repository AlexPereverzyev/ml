from functools import reduce
from classifier import LogisticRegression


class NewtonsClassifier(LogisticRegression):
    """Basic classifier based on ordinary linear regression and sigmoid
    function and Newton - Raphson training algorithm"""

    def batch_train(self, data):
        """Trains using Newton - Raphson batch training algorithm"""
        for k in range(self.iterations):
            theta = self.theta[:]
            for k, t in enumerate(self.theta):
                gradients = []
                hessians = []
                for sample in data:
                    h = self.predict(sample)
                    x = sample[k]
                    xs = reduce(lambda s1, s2: s1 + s2, sample[:-1])
                    y = sample[-1]
                    g = (y - h) * x
                    H = (h * (1 - h)) * x * xs
                    gradients.append(g)
                    hessians.append(H)
                gradient = reduce(lambda x, y: x + y, gradients)
                hessian = reduce(lambda x, y: x + y, hessians)
                if hessian:
                    theta[k] = t + gradient / hessian
                else:
                    continue
            self.theta = theta
