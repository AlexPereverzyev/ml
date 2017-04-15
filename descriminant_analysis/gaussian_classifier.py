import math
from numpy.linalg import det, inv


class GaussianClassifier(object):
    """Classifier based on linear descriminant analysis (LDA)"""

    def __init__(self, classes, dimensions=1):
        if not classes or not dimensions or dimensions < 1:
            raise Exception('Invalid input arguments')
        self.dimensions = dimensions
        self.total = 0
        self.classes = classes
        self.counts = {label: 0 for label in classes}
        self.probs = {label: 0 for label in classes}
        self.means = {label: [0 for _ in range(dimensions)]
                      for label in classes}
        self.cov_matrix = [[0 for _ in range(dimensions)]
                           for __ in range(dimensions)]
        self.cov_matrix_inv = None
        self.scale = 1

    def __str__(self):
        result = ''
        for label in self.classes:
            result += '{0}: {1}\n'.format(label, self.means[label])
        return result

    def train(self, data):
        for sample in data:
            self.total += 1
            label = sample[-1]
            self.counts[label] += 1
            mean = self.means[label]
            self.means[label] = [sample[i] + m for i, m in enumerate(mean)]
        for label in self.classes:
            self.probs[label] = self.counts[label] / self.total
        for label in self.classes:
            count = self.counts[label]
            mean = self.means[label]
            self.means[label] = [m / count for m in mean]
        for sample in data:
            label = sample[-1]
            mean = self.means[label]
            mean_delta = [sample[i] - m for i, m in enumerate(mean)]
            for i in range(self.dimensions):
                for j in range(self.dimensions):
                    c = mean_delta[i] * mean_delta[j]
                    self.cov_matrix[i][j] += c
        self.cov_matrix_inv = inv(self.cov_matrix)
        # self.scale = 1. / (math.pow(2 * math.pi, self.dimensions / 2) *
        #                    math.pow(det(self.cov_matrix), .5))

    def predict(self, data):
        result = []
        for sample in data:
            probs = []
            for label in self.classes:
                mean = self.means[label]
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
                prob = self.scale * math.exp(dot_product)
                probs.append(prob)
            result.append(probs)
        return result
