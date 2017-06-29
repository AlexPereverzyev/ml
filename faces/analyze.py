
from metrics import *
from graphs import *
from data import DataLoader
from persistence import ModelStore
from construction import eigenfaces_from_classifier


model_name = 'pca_svc_2.pkl'
data_path = 'data'
models_path = 'models'
ext, h, w = '.jpg', 70, 70

store = ModelStore(models_path)
samples = DataLoader(data_path, ext)

X, Y, _, _, X_test, Y_test = samples.load().split()
clf = store.load(model_name)

print_model(clf, model_name)
print_score(clf, X_test, Y_test)
print_confusion(clf, X, Y)
# print_mismatches(clf, samples.data, X_test, Y_test)

# plot(eigenfaces_from_classifier(clf, h, w))
# plot_lcurve(clf, X_test, Y_test)
# plot_vcurve(clf, X, Y, 'pca__n_components', range(20, 200, 5))
