
from neural_net.activation_func import *
from neural_net.perceptron import Perceptron


class NeuralLayer(object):
    """Represents set of perceptrons located at certain layer in neural
       network"""
    def __init__(self, dimensions, count=3, is_out=False):
        self.dimensions = dimensions
        self.is_out = is_out
        self.perceptrons = [Perceptron(dimensions, sigmoid)
                            for _ in range(count)]

    def __str__(self):
        return ', '.join(str(p) for p in self.perceptrons)

    def __iter__(self):
        return iter(self.perceptrons)

    def compute(self, sample):
        return [p.compute(sample) for p in self.perceptrons]
