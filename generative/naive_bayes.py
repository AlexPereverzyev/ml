from functools import reduce


class NaiveBayesClassifier(object):
    """Classifier based on naive Bayes method"""
    def __init__(self, classes, dimensions=1):
        if not classes or not dimensions or dimensions < 1:
            raise Exception('Invalid input arguments')
        self.dimensions = dimensions
        self.classes = classes
        self.total = 0
        self.counts = {label: 0 for label in classes}
        self.probs = {label: 0 for label in classes}
        self.theta = {label: [0 for _ in range(dimensions)]
                      for label in classes}

    def __str__(self):
        result = ''
        for i, label in enumerate(self.classes):
            result += '{0} ({1}): {2}'.format(
                label,
                self.probs[label],
                ', '.join(('{0:.4f}'.format(c)
                          for c in self.theta[label])))
            result += '\n' if i < len(self.classes) - 1 else ''
        return result

    def train(self, data):
        for sample in data:
            label = sample[-1]
            self.counts[label] += 1
            self.total += 1
            for i in range(self.dimensions):
                self.theta[label][i] += sample[i]
        for label in self.classes:
            for i in range(self.dimensions):
                self.theta[label][i] += 1
                self.theta[label][i] /= (self.counts[label] + self.dimensions)
            self.probs[label] = self.counts[label] / self.total

    def predict_proba(self, data):
        result = []
        for sample in data:
            probs = []
            probs_xyy = {}
            for label in self.classes:
                prob_y = self.probs[label]
                prob_xy = reduce(lambda a, b: a * b,
                                 (self.theta[label][i]
                                  for i in range(self.dimensions)
                                  if sample[i]))
                probs_xyy[label] = prob_xy * prob_y
            for label in self.classes:
                prob_xyy_cur = probs_xyy[label]
                prob_xyy_rest = [p for lbl, p in probs_xyy.items()
                                 if lbl != label]
                prob_yx = prob_xyy_cur / (
                    prob_xyy_cur + reduce(lambda a, b: a + b, prob_xyy_rest))
                probs.append(prob_yx)
            result.append(probs)
        return result

    def predict(self, data):
        probs = self.predict_proba(data)
        result = []
        for sample_probs in probs:
            label = reduce(lambda p1, p2: p1[0] if p1[1] >= p2[1] else p2[0],
                           ((i, p) for i, p in enumerate(sample_probs)))
            result.append(label)
        return result
