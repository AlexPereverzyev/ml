
import time
import numpy as np
from di import inject
from preprocess import *
from config import AppConfig
from logging import Logger
from sklearn.externals import joblib


class FaceDetector(object):

    threshold = 0.85

    @inject
    def __init__(self,
                 config: AppConfig,
                 logger: Logger):
        self.config = config
        self.logger = logger
        self._clf = None

    @property
    def clf(self):
        if self._clf is None:
            start = time.time()
            self._clf = joblib.load(self.config.model_path)
            self.logger.info('loaded face classifier in: {0:.3f} sec'
                             .format(time.time() - start))
        return self._clf

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
        for s, b in decompose(img_size, step=15, min_size=(40, 40)):
            if scale != s:
                scale = s
                size_scaled = rescale(img_size, scale)
                img_scaled = img.resize(size_scaled)
            frame = img_scaled.crop(b)
            is_face, confidence = self.match(frame)
            if is_face:
                rect = scale_bounds(b, 1 / (pre_scale * s))
                yield confidence, rect, frame

    def find_unique(self, image):
        # todo: check for overlaps instead
        best = (0, None)
        for c, r, _ in self.find_all(image):
            if c > best[0]:
                best = (c, r)
        if best[0]:
            yield best
