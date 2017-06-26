
from metrics import print_model
from construction import *
from data import DataLoader
from learning import HypoSearcher
from persistence import ModelStore

iteration = 2
data_path = 'data'
models_path = 'models'
ext, h, w = '.jpg', 70, 70
param_grid = [{'pca__n_components': [50, 80, 100, 150],
               'svc__kernel': ['rbf'],
               'svc__C': [2, 3, 5, 7, 10],
               'svc__gamma': [0.005, 0.01, 0.015]}]

searcher = HypoSearcher(create_image_classifier())
store = ModelStore(models_path)
samples = DataLoader(data_path, ext)

_, _, X_train, Y_train, _, _ = samples.load().split()
clf = searcher.optimize(X_train, Y_train, param_grid)
store.save(clf, id=iteration)

print_model(clf, 'Best estimator:')
