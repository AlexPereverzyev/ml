
from neural_net.neural_layer import NeuralLayer


class MultiLayerPerceptronClassifier(object):
    """Classifier based on interconnected perceptrons and backpropagation
       learning algorithm"""
    def __init__(self, dimensions, layers=(3,), outputs=1, iterations=100,
                 learning_rate=0.1):
        if (not dimensions or dimensions < 1 or
           not layers or outputs <= 0 or
           not iterations or iterations < 1 or
           not learning_rate or learning_rate < 0):
            raise Exception('Invalid input arguments')
        self.dimensions = dimensions
        self.layers = [NeuralLayer(d, c, (i == len(layers)))
                       for d, c, i in
                       ((layers[i - 1] if i > 0 else dimensions, c, i)
                        for i, c in enumerate(list(layers) + [outputs]))]
        self.learning_rate = learning_rate
        self.iterations = iterations

    def __str__(self):
        return '\n'.join(str(l) for l in self.layers)

    def predict(self, data):
        return [(self.compute(sample)[-1][0]) for sample in data]

    def compute(self, sample):
        outputs = []
        inputs = sample
        for layer in self.layers:
            inputs = layer.compute([1] + inputs)
            outputs.append(inputs)
        return outputs

    def train(self, data):
        for _ in range(self.iterations):
            for sample in data:
                outputs = self.compute(sample)
                errors = [[] for _ in range(len(outputs))]
                for out in outputs[-1]:
                    errors[-1].append(out * (1 - out) * (sample[-1] - out))
                for i in reversed(range(len(outputs[:-1]))):
                    for j, out in enumerate(outputs[i]):
                        e_sum = 0
                        for k, percept in enumerate(self.layers[i + 1]):
                            e_sum += errors[i + 1][k] * percept.theta[j + 1]
                        errors[i].append(out * (1 - out) * e_sum)
                for i, layer in enumerate(self.layers):
                    inputs = outputs[i - 1] if i else sample[:-1]
                    for j, percept in enumerate(layer):
                        error = errors[i][j]
                        for k, t in enumerate(percept.theta):
                            x = inputs[k - 1] if k else 1
                            percept.theta[k] += self.learning_rate * error * x
