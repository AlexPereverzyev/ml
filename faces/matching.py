
import numpy as np


class MatchDetector(object):
    def __init__(self, clf):
        self.clf = clf

    def is_match(self, image):
        sample = np.array(image).ravel()
        r = self.clf.predict([sample])[0]
        return (r > 0)
