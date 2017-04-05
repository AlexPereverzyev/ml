
from data_tools import CorrelatedIterator
from predictor import LinearRegression
from training_sets import *

if __name__ == "__main__":
    details_count = 10

    print('-' * 20, ' Example 1', '-' * 20)
    lr = LinearRegression(1, 100, 0.00001, precision=1)
    with CorrelatedIterator(Training_set1) as feed:
        lr.batch_train(feed)
        print('Et = {0} %'.format(lr.calc_training_error(feed)))
    with CorrelatedIterator(Validation_set1) as feed:
        print('E  = {0} %'.format(lr.calc_training_error(feed)))
        print('-' * 20, 'Details: {0}'.format(lr))
        for d in (d for i, d in enumerate(feed) if i < details_count):
            print('Ye = {0}, Yr = {1}'.format(lr.predict(d), d[-1]))

    print('-' * 20, ' Example 2 ', '-' * 20)
    lr = LinearRegression(11, 10, 0.000000001, precision=50)
    with CorrelatedIterator(Training_set2) as feed:
        lr.batch_train(feed)
        print('Et = {0} %'.format(lr.calc_training_error(feed)))
    with CorrelatedIterator(Validation_set2) as feed:
        print('E  = {0} %'.format(lr.calc_training_error(feed)))
        print('-' * 20, 'Details: {0}'.format(lr))
        for d in (d for i, d in enumerate(feed) if i < details_count):
            print('Ye = {0}, Yr = {1}'.format(lr.predict(d), d[-1]))
