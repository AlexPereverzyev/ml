
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
            priors = self._get_priors_skl()
            source = self._get_theta_skl()
            for i, label in enumerate(self.regression.classes_):
                result += '{0} ({1}): '.format(label, priors[i])
                result += ', '.join(('{0:.4f}'.format(v) for v in source[i]))
                result += '\n' if i < len(self.regression.classes_) - 1 else ''
        print('-' * 20, self.key, '-' * 20)
        print(result)

    def _get_priors_skl(self):
        if hasattr(self.regression, 'class_count_'):
            total = reduce(lambda a, b: a + b, self.regression.class_count_)
            return [count / total for count in self.regression.class_count_]
        if hasattr(self.regression, 'priors_'):
            return self.regression.priors_
        if hasattr(self.regression, 'priors'):
            return self.regression.priors
        raise Exception('The solver is not supported')

    def _get_theta_skl(self):
        if hasattr(self.regression, 'theta_'):
            return self.regression.theta_
        if hasattr(self.regression, 'means_'):
            return self.regression.means_
        if hasattr(self.regression, 'feature_log_prob_'):
            return [[math.exp(l) for l in features]
                    for features in self.regression.feature_log_prob_]
        raise Exception('The solver is not supported')

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
