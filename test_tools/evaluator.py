from functools import reduce


class Evaluator(object):

    def calc_cost(self, predictor, data):
        """Calculates cost function for given predictor and training set"""
        delta = map(lambda d: (predictor.predict(d) - d[-1]) ** 2, data)
        cost = reduce(lambda x, y: x + y, delta) / 2
        return cost

    def calc_score(self, predictor, data, precision=0):
        """Calculates score (1 - training error) for given predictor
           and training set"""
        missed_count = list(map(lambda d:
                            int(abs(predictor.predict(d) - d[-1]) > precision),
                            data))
        misses = reduce(lambda x, y: x + y, missed_count) / len(missed_count)
        return (1 - misses) * 100
