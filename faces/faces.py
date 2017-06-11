import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score


def score(estimator, x):
    return accuracy_score(np.ones(len(x)), estimator.predict(x))

# todo:
# 1. add model persistance
# 2. breakdown the script into modules


data_path = 'data'
data = {}

for face in (f for f in os.listdir(data_path)
             if f.endswith('.jpg') and 'b' not in f):
    key = os.path.splitext(face)[0]
    img = Image.open(os.path.join(data_path, face))
    data[key] = np.array(img)

X = np.array([v.ravel() for k, v in data.items()])
print('Samples : ', len(X))
print('Features: ', len(X[0]))

X_train, X_test = train_test_split(X, test_size=0.25, random_state=1731)

pipe_clf = Pipeline([('reduce', PCA(svd_solver='randomized', whiten=True)),
                     ('match', OneClassSVM())])

param_grid = [{'reduce__n_components': [50, 100, 150],
               'match__kernel': ['linear', 'rbf'],
               'match__nu': [0.005, 0.01, 0.05],
               'match__tol': [0.00005, 0.0001, 0.001, 0.01],
               'match__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01]}]

clf = GridSearchCV(pipe_clf, param_grid, scoring=score).fit(X_train)

print('Best Estimator:')
print(clf.best_estimator_)
print('Train Score: ', score(clf, X_train))
print('Test  Score: ', score(clf, X_test))


# def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
#     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
#     plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
#     for i in range(n_row * n_col):
#         plt.subplot(n_row, n_col, i + 1)
#         plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
#         plt.title(titles[i], size=12)
#         plt.xticks(())
#         plt.yticks(())


# h, w = 70, 70
# pca = clf.best_estimator_.named_steps['reduce']
# eigenfaces = pca.components_.reshape((pca.n_components, h, w))
# titles = ['eigenface {0}'.format(i + 1) for i in range(eigenfaces.shape[0])]
# plot_gallery(eigenfaces, titles, h, w)
# plt.show()
