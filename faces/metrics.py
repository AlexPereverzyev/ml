
def print_score(clf, x, y, label='Score: '):
    print(label, clf.score(x, y))


def print_scores(clf, x_train, y_train, x_test, y_test):
    print_score(clf, x_train, y_train, 'Train Score : ')
    print_score(clf, x_test, y_test, 'Test Score  : ')


def print_model(clf, label):
    print(label)
    print(clf.get_params(deep=False))


def print_mismatches(clf, data, x, y, ext):
    Y = clf.predict(x)
    P = clf.predict_proba(x)
    mismatches = ['{0:.2f}:{1:.2f} - {2}{3}'.format(*c_p, f, ext)
                  for f, y_r, y_p, c_p in zip(data.keys(), Y, y, P)
                  if y_r != y_p]
    print('Mismatches:')
    for m in mismatches:
        print(m)
