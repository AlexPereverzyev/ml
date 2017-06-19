
from sklearn.metrics import confusion_matrix


def print_score(clf, x, y, label='Score: '):
    print(label, clf.score(x, y))


def print_scores(clf, x_train, y_train, x_test, y_test):
    print_score(clf, x_train, y_train, 'Train Score : ')
    print_score(clf, x_test, y_test, 'Test Score  : ')


def print_model(clf, label):
    print(label)
    print(clf.get_params(deep=False))


def print_mismatches(clf, data, x, y):
    Y = clf.predict(x)
    P = clf.predict_proba(x)
    F = [_map_to_file(data, s) for s in x]
    mismatches = ['{0:.2f}:{1:.2f} - {2}'.format(*c_p, f)
                  for f, y_r, y_p, c_p in zip(F, Y, y, P)
                  if y_r != y_p]
    print('Mismatches:')
    for m in mismatches:
        print(m)


def print_confusion(clf, x, y):
    confusion = confusion_matrix(y, clf.predict(x))
    print('Confusion Matrix:')
    print(confusion)


def _map_to_file(data, x):
    for k, v in data.items():
        for s1, s2 in zip(v.ravel(), x):
            if s1 != s2:
                break
        else:
            return k
