
from test_tools.stats import Stats


class RegressionTester(object):
    """Wraps learning algorithm and provides number of helper methods
       to train, evaluate and compare different regression models"""

    def __init__(self, key, regression, data_cache, precision=.1):
        if not key or not regression or data_cache is None:
            raise Exception('Invalid constructor arguments')
        self.key = key
        self.regression = regression
        self.precision = precision
        self.data_cache = data_cache

    @property
    def is_skl(self):
        return not hasattr(self.regression, 'theta')

    def train(self, dataset):
        data = self.data_cache[dataset]
        if not self.is_skl:
            self.regression.train(data)
        else:
            X, Y = self.split(data)
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
        data = self.data_cache[dataset]
        X, Y = self.split(data)
        print('S = {0:.2f} %  C = {1:.2f}'
              .format(*Stats().calculate(
                  self.regression, X, Y, self.precision)))

    def printy(self, dataset, count=10):
        if not count:
            return
        data = self.data_cache[dataset]
        X, Y = self.split(data)
        for i, y in ((i, y) for i, y in
                     enumerate(self.regression.predict(X)) if i < count):
            print('Ye = {0:.2f}, Yr = {1}'.format(y, Y[i]))

    def split(self, data):
        lt = int(self.is_skl)
        X = [f[lt:-1] for f in data]
        Y = [f[-1] for f in data]
        return X, Y
