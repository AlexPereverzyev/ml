
import random
from functools import reduce
from neural_net.activation_func import perceptron


class Perceptron(object):
    """Basic building block of neural network: has number of outputs,
       their parameters and activation function"""
    def __init__(self, dimensions=1, activation=perceptron):
        self.theta = [random.uniform(-0.05, 0.05)
                      for _ in range(dimensions + 1)]
        self.activation = activation

    def __str__(self):
        result = '[ {0:.8f} ] ('.format(self.theta[0])
        result += '  '.join(['{0:.8f}'.format(t) for t in self.theta[1:]])
        result += ')'
        return result

    def compute(self, sample):
        product = reduce(lambda p1, p2: p1 + p2,
                         (t * sample[i] for i, t in enumerate(self.theta)))
        return self.activation(product)
