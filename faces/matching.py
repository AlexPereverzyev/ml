
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
        scale = .0
        for i, (s, b) in enumerate(decompose(img_size, step=15)):
            if scale != s:
                scale = s
                size_scaled = rescale(img_size, scale)
                img_scaled = img.resize(size_scaled)
            frame = img_scaled.crop(b)
            is_face, confidence = self.match(frame)
            if is_face:
                rect = scale_bounds(b, 1 / (pre_scale * s))
                yield confidence, rect, frame
