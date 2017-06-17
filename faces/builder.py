

from metrics import *
from graphs import *
from construction import *
from data import DataLoader
from learning import HypoSearcher
from persistence import ModelStore


models_path = 'models'
data_path = 'data'
ext, h, w = '.jpg', 70, 70
param_grid = [{'pca__n_components': [40, 50, 80],
               'svc__kernel': ['rbf'],
               'svc__C': [1, 2, 3, 5, 10],
               'svc__tol': [0.00005, 0.0001, 0.001, 0.01],
               'svc__gamma': [0.001, 0.005, 0.01]}]

searcher = HypoSearcher(create_image_classifier())
store = ModelStore(models_path)
samples = DataLoader(data_path, ext)
samples.mask = 'b'

X, Y, _, _, _, _ = samples.load().split()

# clf = searcher.optimize(X, Y, param_grid)
# store.save(clf)
# print_model(clf, 'Best estimator:')

# plot(eigenfaces_from_classifier(clf, h, w))
# plot_lcurve(clf, X, Y)
# plot_vcurve(clf, X, Y, 'pca__n_components', [25, 40, 50, 80, 100])

model_name = 'pca_svc_20170617-211110.pkl'
clf = store.load(model_name)

# print_model(clf, model_name)
print_score(clf, X, Y)
print_mismatches(clf, samples.data, X, Y, ext)
# priny_confusion(clf, X, Y)
