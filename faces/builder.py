

from metrics import *
from graphs import plot
from construction import *
from data import DataLoader
from learning import HypoSearcher
from persistence import ModelStore


models_path = 'models'
data_path = 'data'
ext, h, w = '.jpg', 70, 70
param_grid = [{'pca__n_components': [40, 50, 80],
               'svc__kernel': ['rbf'],
               'svc__C': [1, 10, 100],
               'svc__tol': [0.00005, 0.0001, 0.001, 0.01],
               'svc__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01]}]

data_loader = DataLoader(data_path, ext)
data_loader.mask = 'nm'
model_store = ModelStore(models_path)
searcher = HypoSearcher(create_image_classifier())

X, Y, X_train, Y_train, X_test, Y_test = data_loader.load().split()

# clf = searcher.optimize(X_train, Y_train, param_grid)
# model_store.save(clf)
model_name = 'pca_svc_n40_c10.pkl'
clf = model_store.load(model_name)

# print_model(clf, 'Best estimator:')
# plot(eigenfaces_from_classifier(clf, h, w))

print_model(clf, model_name)
print_scores(clf, X_train, Y_train, X_test, Y_test)
print_mismatches(clf, data_loader.data, X, Y, ext)
