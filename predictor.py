from functools import reduce


class LinearRegression(object):
    """Basic value predictor based on ordinary linear regression"""

    def __init__(self, dimensions=1, iterations=100, learning_rate=0.00001,
                 init_theta=.0):
        if (not dimensions or dimensions < 1 or
           not iterations or iterations < 1 or
           not learning_rate or learning_rate < 0 or
           init_theta is None):
            raise Exception('Invalid input arguments')
        self.theta = [init_theta for _ in range(dimensions + 1)]
        self.iterations = iterations
        self.learning_rate = learning_rate

    def __str__(self):
        result = 'Y = '
        if not self.theta:
            result += '?'
            return result
        for i, t in enumerate(self.theta):
            sign = '+' if (i != 0 and t >= 0) else ''
            result += '{0}{1}*X{2} '.format(sign, t, i)
        return result

    def predict(self, sample):
        """Evaluates target function using pre-trained parameters
        and provided sample"""
        if not self.theta:
            raise Exception('Linear regression should be trained first')
        inputs = [1] + sample
        result = reduce(lambda p1, p2: p1 + p2,
                        (t * inputs[i] for i, t in enumerate(self.theta)))
        return result

    def calc_cost(self, data_feed):
        """Calculates cost function for given training set and parameters"""
        if not self.theta:
            raise Exception('Linear regression should be trained first')
        delta = map(lambda d: (self.predict(d) - d[-1]) ** 2, data_feed)
        cost = reduce(lambda x, y: x + y, delta) / 2
        return cost

    def calc_training_error(self, data_feed):
        pass

    def calc_learning_rate(self, cur, rlen):
        """Calculates new value of learning rate based on the current one
        and number of iterations remaining"""
        return cur * (1. - rlen / self.iterations)

    def batch_train(self, data_feed):
        """Trains using Least Mean Squares batch training algorithm"""
        learning_rate = self.learning_rate
        for k in range(self.iterations):
            theta = self.theta[:]
            for i, t in enumerate(self.theta):
                sample_gradients = []
                for sample in data_feed:
                    inputs = [1] + sample
                    x = inputs[i]
                    y = sample[-1]
                    h = self.predict(sample)
                    g = (y - h) * x
                    sample_gradients.append(g)
                gradient = reduce(lambda x, y: x + y, sample_gradients)
                theta[i] = t + learning_rate * gradient
            self.theta = theta
            learning_rate = self.calc_learning_rate(learning_rate, k + 1)

    def stohastic_train(self, sample):
        """Trains using Least Mean Squares stohastic training algorithm"""
        # todo
        pass
