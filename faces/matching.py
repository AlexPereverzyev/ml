
import numpy as np


class MatchDetector(object):

    threshold = 0.9

    def __init__(self, clf):
        self.clf = clf

    def match(self, image):
        sample = np.array(image).ravel()
        probs = self.clf.predict_proba([sample])[0]
        r = probs[1] > probs[0] and probs[1] > self.threshold
        p = probs[int(r)]
        return r, p
