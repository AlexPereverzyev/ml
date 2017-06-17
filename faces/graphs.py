
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve, learning_curve


def plot(images, n_row=3, n_col=4):
    _, h, w = images.shape
    plt.figure(figsize=(2 * n_col, 2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    plt.show()


def plot_vcurve(clf, x, y, param, values):
    nvalues = np.array(values)
    train_scores, test_scores = validation_curve(clf, x, y, param, nvalues)
    plt.title('Validation Curve for \'{0}\': {1}'
              .format(param, ', '.join([str(v) for v in values])))
    plt.plot(nvalues, train_scores, '-r', linewidth=1, label='Train')
    plt.plot(nvalues, test_scores, '-b', linewidth=1, label='Test')
    plt.legend()
    plt.show()


def plot_lcurve(clf, x, y):
    train_sizes, train_scores, test_scores = learning_curve(clf, x, y)
    plt.title('Learning Curve with {0} samples'
              .format(', '.join([str(s) for s in train_sizes])))
    plt.plot(train_sizes, train_scores, '-r', linewidth=1, label='Train')
    plt.plot(train_sizes, test_scores, '-b', linewidth=1, label='Test')
    plt.legend()
    plt.show()
