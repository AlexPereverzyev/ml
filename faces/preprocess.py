
from PIL import Image


class ImageSplitter(object):
    def __init__(self, frame=(70, 70)):
        self.frame = frame

    def split(self, image):
        # todo
        # 1. split image into 70x70 parts
        # 2. scale up/down and repeat
        for i in range(10):
            yield [1]
