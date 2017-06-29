
import numpy as np
from preprocess import *


class MatchDetector(object):

    threshold = 0.85

    def __init__(self, clf):
        self.clf = clf

    def match(self, image):
        sample = np.array(image).ravel()
        probs = self.clf.predict_proba([sample])[0]
        r = probs[1] > probs[0] and probs[1] > self.threshold
        p = probs[int(r)]
        return r, p

    def find_all(self, image):
        pre_scale = prescale(image.size)
        img_size = rescale(image.size, pre_scale)
        img = image.resize(img_size)
        img = img.convert('L')
        for i, (s, b) in enumerate(decompose(img_size, step=15)):
            region = img.resize(rescale(img_size, s)).crop(b)
            is_face, confidence = self.match(region)
            if is_face:
                yield confidence, scale_bounds(b, 1 / (pre_scale * s)), region
