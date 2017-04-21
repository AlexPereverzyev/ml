import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm
from data_tools.data_iter import DataIterator


class DataPlotter(object):
    """Data visualization helper"""

    @staticmethod
    def plot_all(dataset, ignore=[], plot_y=True):
        with DataIterator(dataset) as data_generator:
            data = list(data_generator)
        X = [f[:-1] for f in data]
        count = len(X[0])
        colors = cm.rainbow(np.linspace(0, 1, count + 1))
        for i in range(count):
            if i in ignore:
                continue
            transposed = []
            for sample in X:
                transposed.append(sample[i])
            pyplot.plot(transposed, color=colors[i])
        if plot_y:
            Y = [f[-1] for f in data]
            pyplot.plot(Y, 'g^', color=colors[-1])
        pyplot.show()
