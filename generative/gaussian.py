import math
from numpy.linalg import inv
from generative.naive_bayes import NaiveBayesClassifier
from functools import reduce


class GaussianClassifier(NaiveBayesClassifier):
    """Classifier based on Gaussian distribution (buggy)"""

    def __init__(self, classes, dimensions=1):
        super().__init__(classes, dimensions)
        self.cov_matrix = [[0. for _ in range(dimensions)]
                           for __ in range(dimensions)]
        self.cov_matrix_inv = None

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
            mean = self.theta[label]
            delta = [sample[i] - m for i, m in enumerate(mean)]
            for i in range(self.dimensions):
                for j in range(self.dimensions):
                    self.cov_matrix[i][j] += delta[i] * delta[j] / self.total
        self.cov_matrix_inv = inv(self.cov_matrix)

    def predict_proba(self, data):
        result = []
        for sample in data:
            probs = []
            for label in self.classes:
                mean = self.theta[label]
                delta = [sample[i] - m for i, m in enumerate(mean)]
                row_vector = [0 for _ in range(self.dimensions)]
                for i in range(self.dimensions):
                    for j in range(self.dimensions):
                        row_vector[i] += self.cov_matrix_inv[j][i] * delta[j]
                dot_product = reduce(lambda x, y: x + y,
                                     (-.5 * row_vector[i] * delta[i]
                                      for i in range(self.dimensions)))
                prob = math.exp(dot_product)
                probs.append(prob)
            result.append(probs)
        return result
