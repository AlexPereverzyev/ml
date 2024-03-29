from functools import reduce


class LinearRegression(object):
    """Basic value predictor based on ordinary linear regression"""

    def __init__(self, dimensions=1, iterations=100, learning_rate=0.00001):
        if (not dimensions or dimensions < 1 or
           not iterations or iterations < 1 or
           not learning_rate or learning_rate < 0):
            raise Exception('Invalid input arguments')
        self.theta = [0 for _ in range(dimensions + 1)]
        self.iterations = iterations
        self.learning_rate = learning_rate

    def __str__(self):
        result = '[ {0:.8f} ] [[ '.format(self.theta[0])
        result += '  '.join(['{0:.8f}'.format(t) for t in self.theta[1:]])
        result += ' ]]'
        return result

    def calc_learning_rate(self, cur, rlen):
        """Calculates new value of learning rate based on the current one
        and number of iterations remaining"""
        return cur * (1. - rlen / self.iterations)

    def train(self, data):
        """Trains using Least Mean Squares batch training algorithm"""
        learning_rate = self.learning_rate
        for k in range(self.iterations):
            theta = self.theta[:]
            for i, t in enumerate(self.theta):
                gradients = []
                for sample in data:
                    h = self.predict([sample])[0]
                    x = sample[i]
                    y = sample[-1]
                    g = (y - h) * x
                    gradients.append(g)
                gradient = reduce(lambda x, y: x + y, gradients)
                theta[i] = t + learning_rate * gradient
            self.theta = theta
            learning_rate = self.calc_learning_rate(learning_rate, k + 1)

    def stohastic_train(self, sample):
        """Trains using Least Mean Squares stohastic training algorithm"""
        theta = self.theta[:]
        for i, t in enumerate(self.theta):
            x = sample[i]
            y = sample[-1]
            h = self.predict([sample])[0]
            gradient = (y - h) * x
            theta[i] = t + self.learning_rate * gradient
        self.theta = theta

    def predict(self, data):
        """Evaluates target function using pre-trained parameters
        and provided sample"""
        if not self.theta:
            raise Exception('Regression should be trained first')
        result = []
        for sample in data:
            result.append(reduce(lambda p1, p2: p1 + p2,
                          (t * sample[i] for i, t in enumerate(self.theta))))
        return result
