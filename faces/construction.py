
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA


def create_image_classifier():
    clf_pipe = Pipeline([
        ('pca', PCA(svd_solver='randomized', whiten=True)),
        ('svc', SVC(probability=True))])
    return clf_pipe


def eigenfaces_from_classifier(clf, h, w):
    pca = clf.named_steps['pca']
    eigenfaces = pca.components_.reshape((pca.n_components, h, w))
    return eigenfaces
