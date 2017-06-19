
import os
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.model_selection import train_test_split


class DataLoader(object):
    def __init__(self, data_path, ext):
        self.data_path = data_path
        self.data = OrderedDict()
        self.state = 57319
        self.mask = None
        self.ext = ext

    def load(self):
        for face in (f for f in os.listdir(self.data_path)
                     if f.endswith(self.ext) and
                     (self.mask is None or
                      all(m not in f for m in self.mask))):
            img = Image.open(os.path.join(self.data_path, face))
            self.data[face] = np.array(img)
        return self

    def split(self, print_stats=True):
        X, Y = [], []
        for k, v in self.data.items():
            X.append(v.ravel())
            Y.append(int('n' not in k))
        X, Y = np.array(X), np.array(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.25, random_state=self.state)
        if print_stats:
            print('Samples   : ', len(X))
            print('Features  : ', len(X[0]))
            print('Train Set : ', len(X_train))
            print('Test Set  : ', len(X_test))
        return X, Y, X_train, Y_train, X_test, Y_test
