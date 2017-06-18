
from metrics import print_model
from construction import *
from data import DataLoader
from learning import HypoSearcher
from persistence import ModelStore


models_path = 'models'
data_path = 'data'
ext, h, w = '.jpg', 70, 70
# param_grid = [{'pca__n_components': [40, 50, 80],
#                'svc__kernel': ['rbf'],
#                'svc__C': [1, 2, 3, 5, 10],
#                'svc__tol': [0.00005, 0.0001, 0.001, 0.01],
#                'svc__gamma': [0.001, 0.005, 0.01, 0.05, 0.1]}]
param_grid = [{'pca__n_components': range(20, 200, 5),
               'svc__kernel': ['rbf'],
               'svc__C': range(1, 10, 1),
               'svc__gamma': [0.01, 0.05, 0.1]}]

searcher = HypoSearcher(create_image_classifier())
store = ModelStore(models_path)
samples = DataLoader(data_path, ext)

X, Y, _, _, _, _ = samples.load().split()
clf = searcher.optimize(X, Y, param_grid)
store.save(clf)

print_model(clf, 'Best estimator:')
