
from data_tools import CorrelatedIterator
from predictor import LinearRegression

_training_set1 = """
        test_data/basic.csv|@N,one,X,X^2,X^3,Y|:6"""
_validation_set1 = """
        test_data/basic.csv|@N,one,X,X^2,X^3,Y"""

_training_set2 = """
        test_data/Calls Per Minute.csv|@date,one,value|:6;
        test_data/Errors Per Minute.csv|@date,value;
        test_data/Average Response Time.csv|@date,value;"""
_validation_set2 = """
        test_data/Calls Per Minute.csv|@date,one,value;
        test_data/Errors Per Minute.csv|@date,value;
        test_data/Average Response Time.csv|@date,value;"""

if __name__ == "__main__":
    details_count = 10

    print('-' * 20, ' Example 1 ', '-' * 20)
    lr = LinearRegression(1, 1000, 0.00001, precision=1)
    with CorrelatedIterator(_training_set1) as feed:
        lr.batch_train(feed)
        print('Et = {0}'.format(lr.calc_training_error(feed)))
    with CorrelatedIterator(_validation_set1) as feed:
        print('E  = {0}'.format(lr.calc_training_error(feed)))
        print('-' * 20, 'Details: {0}'.format(lr))
        for d in (d for i, d in enumerate(feed) if i < details_count):
            print('Ye = {0}, Yr = {1}'.format(lr.predict(d), d[-1]))

    print('-' * 20, ' Example 2 ', '-' * 20)
    lr = LinearRegression(2, 300, 0.0000001, 5, precision=50)
    with CorrelatedIterator(_training_set2) as feed:
        lr.batch_train(feed)
        print('Et = {0} %'.format(lr.calc_training_error(feed)))
    with CorrelatedIterator(_validation_set2) as feed:
        print('E  = {0} %'.format(lr.calc_training_error(feed)))
        print('-' * 20, 'Details: {0}'.format(lr))
        for d in (d for i, d in enumerate(feed) if i < details_count):
            print('Ye = {0}, Yr = {1}'.format(lr.predict(d), d[-1]))
