from data_tools.data_iter import DataIterator
from test_tools.stats import Stats


class RegressionTester(object):
    """Wraps learning algorithm and provides number of helper methods
       to train, evaluate and compare different regression models"""

    def __init__(self, key, regression, precision=.1):
        if not key or not regression:
            raise Exception('Regression key or instance not specified')
        self.key = key
        self.regression = regression
        self.precision = precision

    @property
    def is_skl(self):
        return not hasattr(self.regression, 'theta')

    def train(self, dataset):
        with DataIterator(dataset) as data_generator:
            data = list(data_generator)
        if not self.is_skl:
            self.regression.train(data)
        else:
            X = [f[1:-1] for f in data]
            Y = [f[-1] for f in data]
            self.regression.fit(X, Y)

    def printr(self):
        rstring = ''
        if not self.is_skl:
            rstring = str(self.regression)
        else:
            rstring = '{0} {1}'.format(
                self.regression.intercept_,
                str(self.regression.coef_).replace('\n', ''))
        print('-' * 20, self.key, '-' * 20)
        print('P = ', rstring)

    def prints(self, dataset):
        with DataIterator(dataset) as data_generator:
            data = list(data_generator)
        lt = int(self.is_skl)
        X = [f[lt:-1] for f in data]
        Y = [f[-1] for f in data]
        print('S = {0:.2f} %  C = {1:.2f}'
              .format(*Stats().calculate(
                  self.regression, X, Y, self.precision)))

    def printy(self, dataset, count=10):
        if not count:
            return
        with DataIterator(dataset) as data_generator:
            data = list(data_generator)
        lt = int(self.is_skl)
        X = [f[lt:-1] for f in data]
        Y = [f[-1] for f in data]
        for i, y in ((i, y) for i, y in
                     enumerate(self.regression.predict(X)) if i < count):
            print('Ye = {0:.2f}, Yr = {1}'.format(y, Y[i]))
