
import math
from functools import reduce
from test_tools.stats import Stats
from test_tools.regression_tester import RegressionTester


class GenerativeTester(RegressionTester):
    """Tester adopted for generative algorithms"""
    def printr(self):
        result = ''
        if not self.is_skl:
            result = str(self.regression)
        else:
            count = reduce(lambda a, b: a + b, self.regression.class_count_)
            for i, label in enumerate(self.regression.classes_):
                result += '{0} ({1}): '.format(
                    label,
                    self.regression.class_count_[i] / count)
                if hasattr(self.regression, 'theta_'):
                    source = self.regression.theta_[i]
                    func = lambda x: x
                else:
                    source = self.regression.feature_log_prob_[i]
                    func = lambda x: math.exp(x)
                result += ', '.join(
                    ('{0:.4f}'.format(func(v)) for v in source))
                result += '\n' if i < len(self.regression.classes_) - 1 else ''
        print('-' * 20, self.key, '-' * 20)
        print(result)

    def prints(self, dataset):
        data = self.data_cache[dataset]
        X, Y = self.split(data)
        print('S = {0:.2f} %'
              .format(Stats().calculate_strict(
                  self.regression, X, Y)))

    def printy(self, dataset, count=10):
        if not count:
            return
        data = self.data_cache[dataset]
        X, Y = self.split(data)
        for i, ps in ((i, ps) for i, ps in
                      enumerate(self.regression.predict_proba(X))
                      if i < count):
            probs = ', '.join(('{0:.4f}'.format(p) for p in ps))
            print('L: {0}, Label: {1}'.format(probs, Y[i]))

    def split(self, data):
        X = [f[:-1] for f in data]
        Y = [f[-1] for f in data]
        return X, Y
