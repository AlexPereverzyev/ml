from functools import reduce


class Stats(object):
    """Basic hypothesis evaluator class which calculates score,
       cost function etc."""

    def calculate(self, predictor, X, Y, precision=0):
        """Calculates agregate predictor stats on the given data set"""
        stats = [(int(abs(p - y) <= precision),  # success
                 (p - y) ** 2)                   # cost
                 for y, p in ((Y[i], p) for i, p in
                 enumerate(predictor.predict(X)))]
        score = reduce(lambda x, y: x + y, (s for s, c in stats)) / len(stats)
        cost = reduce(lambda x, y: x + y, (c for s, c in stats)) / 2
        return (score * 100, cost)

    def calculate_strict(self, predictor, X, Y):
        """Calculates strict predictor score on the given data set"""
        stats = [int(p == y)                    # success
                 for y, p in ((Y[i], p) for i, p in
                 enumerate(predictor.predict(X)))]
        score = reduce(lambda x, y: x + y, stats) / len(stats)
        return score * 100
