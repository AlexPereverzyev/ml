import math
from numpy.linalg import inv
from generative.naive_bayes import NaiveBayesClassifier
from functools import reduce


class GaussianClassifier(NaiveBayesClassifier):
    """Classifier based on plain Gaussian distribution"""

    def __init__(self, classes, dimensions=1):
        super().__init__(classes, dimensions)
        self.cov_matrix = {label: [[0 for _ in range(dimensions)]
                           for __ in range(dimensions)]
                           for label in self.classes}
        self.cov_matrix_inv = {label: None for label in self.classes}

    def train(self, data):
        for sample in data:
            self.total += 1
            label = sample[-1]
            self.counts[label] += 1
            mean = self.theta[label]
            self.theta[label] = [sample[i] + m for i, m in enumerate(mean)]
        for label in self.classes:
            self.probs[label] = self.counts[label] / self.total
        for label in self.classes:
            count = self.counts[label]
            mean = self.theta[label]
            self.theta[label] = [m / count for m in mean]
        for sample in data:
            label = sample[-1]
            prob = self.probs[label]
            mean = self.theta[label]
            cov_matrix = self.cov_matrix[label]
            delta = [sample[i] - m for i, m in enumerate(mean)]
            for i in range(self.dimensions):
                for j in range(self.dimensions):
                    cov_matrix[i][j] += prob * delta[i] * delta[j]
        for label in self.classes:
            cov_matrix = self.cov_matrix[label]
            self.cov_matrix_inv[label] = inv(cov_matrix)

    def predict_proba(self, data):
        result = []
        for sample in data:
            probs = []
            for label in self.classes:
                mean = self.theta[label]
                cov_matrix_inv = self.cov_matrix_inv[label]
                delta = [sample[i] - m for i, m in enumerate(mean)]
                v = [0 for _ in range(self.dimensions)]
                for i in range(self.dimensions):
                    for j in range(self.dimensions):
                        v[i] += cov_matrix_inv[i][j] * delta[j]
                dot_product = reduce(lambda x, y: x + y,
                                     (-.5 * v[i] * delta[i]
                                      for i in range(self.dimensions)))
                prob = math.exp(dot_product)
                probs.append(prob)
            result.append(probs)
        return result
