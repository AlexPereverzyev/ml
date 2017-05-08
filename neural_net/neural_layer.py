
from neural_net.perceptron import Perceptron


class NeuralLayer(object):
    """Represents set of perceptrons located at certain layer in neural
       network"""
    def __init__(self, dimensions, count=3):
        self.dimensions = dimensions
        self.perceptrons = [Perceptron(dimensions) for _ in range(count)]

    def __str__(self):
        return ', '.join(str(p) for p in self.perceptrons)

    def compute(self, sample):
        return [p.compute(sample) for p in self.perceptrons]
