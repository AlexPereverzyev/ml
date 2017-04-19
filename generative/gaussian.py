import math
from numpy.linalg import inv
from generative.naive_bayes import NaiveBayesClassifier


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
            mean_delta = [sample[i] - m for i, m in enumerate(mean)]
            for i in range(self.dimensions):
                for j in range(self.dimensions):
                    c = mean_delta[i] * mean_delta[j] / self.total
                    self.cov_matrix[i][j] += c
        self.cov_matrix_inv = inv(self.cov_matrix)

    def predict_proba(self, data):
        result = []
        for sample in data:
            probs = []
            for label in self.classes:
                mean = self.theta[label]
                mean_delta = [sample[i] - m for i, m in enumerate(mean)]
                cov_vector = []
                for i in range(self.dimensions):
                    v = 0
                    for j in range(self.dimensions):
                        v += self.cov_matrix_inv[i][j] * mean_delta[j]
                    cov_vector.append(v)
                dot_product = 0
                for i in range(self.dimensions):
                    dot_product += -.5 * cov_vector[i] * mean_delta[i]
                prob = math.exp(dot_product)
                probs.append(prob)
            result.append(probs)
        return result
