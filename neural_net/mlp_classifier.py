
from neural_net.activation_func import sigmoid
from neural_net.neural_layer import NeuralLayer


class MultiLayerPerceptronClassifier(object):
    """Classifier based on interconnected perceptrons and backpropagation
       learning algorithm"""
    def __init__(self, dimensions, layers=(3,), iterations=100,
                 learning_rate=0.01):
        if (not dimensions or dimensions < 1 or
           not layers or
           not iterations or iterations < 1 or
           not learning_rate or learning_rate < 0):
            raise Exception('Invalid input arguments')
        self.dimensions = dimensions
        self.layers = [NeuralLayer(d, c) for d, c in
                       ((layers[i - 1] if i > 0 else dimensions, l)
                        for i, l in enumerate(list(layers) + [1]))]
        self.layers[-1].perceptrons[0].activation = sigmoid
        self.learning_rate = learning_rate
        self.iterations = iterations

    def __str__(self):
        result = '\n'.join(str(l) for l in self.layers)
        return result

    def predict(self, data):
        result = []
        for sample in data:
            result.append(self.predcit_single(sample))
        return result

    def predcit_single(self, sample):
        result = sample
        for layer in self.layers:
            result = layer.compute([1] + result)
        return result

    def train(self, data):
        # todo
        for sample in data:
            pass
