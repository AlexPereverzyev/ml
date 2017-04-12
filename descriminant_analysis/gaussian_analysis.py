
class GaussianDescriminantAnalysis(object):
    """Classifier based on Gaussian (linear) descriminant analysis"""

    def __init__(self, dimensions=1, classes=2, precision=10):
        if not dimensions or dimensions < 1:
            raise Exception('Invalid input arguments')
        self.probs = [0 for _ in range(classes)]
        self.means = [[0 for __ in range(dimensions)] for _ in range(classes)]
        self.covariance = 0

    def __str__(self):
        return ''

    def train(self, data):
        pass

    def predict(self, sample):
        pass
