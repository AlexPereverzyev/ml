
from sklearn.model_selection import GridSearchCV


class HypoSearcher(object):
    def __init__(self, clf):
        self.clf = clf

    def optimize(self, x, y, params):
        clf_search = GridSearchCV(self.clf, params, verbose=100).fit(x, y)
        clf = clf_search.best_estimator_
        return clf
