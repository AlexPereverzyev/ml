
class LinearRegression(object):
    """TODO"""
    def __init__(self):
        # test
        self.theta = [1, 0.5, -0.25, 0.125]

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

        result = .0

        for i, x in enumerate(sample):
            result += self.theta[i] * x

        return result

    def batch_train(self, data_iter):
        # todo
        pass

    def stohastic_train(self, data_sample):
        # todo
        pass
