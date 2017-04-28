import numpy as np
from scipy import linalg
from generative.gaussian import GaussianClassifier


class LinearDescriminantClassifier(GaussianClassifier):
    """Classifier and data transformer based on linear descriminant analysis"""
    def __init__(self, classes, dimensions=1):
        super().__init__(classes, dimensions)
        self.common_mean = [0 for _ in range(dimensions)]
        self.scatter_wi = [[0 for _ in range(dimensions)]
                           for __ in range(dimensions)]
        self.scatter_bw = [[0 for _ in range(dimensions)]
                           for __ in range(dimensions)]
        self.scatter_wi_inv = None
        self.transform_mx = [[0 for _ in range(dimensions)]
                             for __ in range(dimensions)]
        self.sigma = [[0 for _ in range(self.dimensions)]
                      for __ in range(self.dimensions)]
        self.bias = 0

    def train(self, data):
        super().train(data)
        for label in self.classes:
            prob = self.probs[label]
            cov_matrix = self.cov_matrix[label]
            for i in range(self.dimensions):
                for j in range(self.dimensions):
                    self.scatter_wi[i][j] += prob * cov_matrix[i][j]
        for label in self.classes:
            prob = self.probs[label]
            mean = self.theta[label]
            for i, m in enumerate(mean):
                self.common_mean[i] += prob * m
        for label in self.classes:
            mean = self.theta[label]
            delta = [self.common_mean[i] - m for i, m in enumerate(mean)]
            for i in range(self.dimensions):
                for j in range(self.dimensions):
                    self.scatter_bw[i][j] += delta[i] * delta[j]
        evals, evecs = linalg.eigh(self.scatter_bw, self.scatter_wi)
        evecs = evecs[:, np.argsort(evals)[::-1]]
        evecs /= np.apply_along_axis(np.linalg.norm, 0, evecs)
        self.transform_mx = evecs
        means = [self.theta[lbl] for lbl in self.classes]
        means_evecs = [[0 for _ in range(self.dimensions)]
                       for __ in range(self.dimensions)]
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                for k in range(self.dimensions):
                    means_evecs[i][j] += means[i][k] * evecs[k][j]
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                for k in range(self.dimensions):
                    self.sigma[i][j] += means_evecs[i][k] * evecs[j][k]
        # todo: refactor
        self.bias = (-.5 * np.diag(np.dot(means, np.array(self.sigma).T)) +
                     np.log([self.probs[lbl] for lbl in self.classes]))

    def transform(self, data):
        new_data = []
        new_dimensions = len(self.classes) - 1
        for sample in data:
            d = [0 for _ in range(new_dimensions)]
            for i in range(new_dimensions):
                for j in range(self.dimensions):
                    d[i] += sample[j] * self.transform_mx[j][i]
            new_data.append(d)
        return new_data

    def predict(self, data):
        scores = []
        for sample in data:
            score = [0 for _ in range(self.dimensions)]
            for i in range(self.dimensions):
                for j in range(self.dimensions):
                    score[i] += sample[j] * self.sigma[i][j]
            for i, _ in enumerate(score):
                score[i] += self.bias[i]
            scores.append(score)
        pred = []
        for score in scores:
            label = self.classes[score.index(max(score))]
            pred.append(label)
        return pred

    def predict_proba(self, data):
        # todo: use sigmoid and score as exp argument
        raise Exception('not supported yet')
