
from data_tools import CorrelatedIterator
from predictor import LinearRegression

_training_set1 = """test_data/basic.csv|@N,one,X,Y|:6"""
_validation_set1 = """test_data/basic.csv|@N,one,X,Y|7:"""

_training_set2 = """
        test_data/Calls Per Minute.csv|@date,one,value|:7;
        test_data/Errors Per Minute.csv|@date,value;
        test_data/Average Response Time.csv|@date,value;"""
_validation_set2 = """
        test_data/Calls Per Minute.csv|@date,one,value|8:;
        test_data/Errors Per Minute.csv|@date,value;
        test_data/Average Response Time.csv|@date,value;"""

if __name__ == "__main__":
    print('----------- example 1 ------------')
    lr = LinearRegression(1, 10, 0.00001)
    with CorrelatedIterator(_training_set1) as feed:
        lr.batch_train(feed)
        print('Ct = {0}'.format(lr.calc_cost(feed)))
    with CorrelatedIterator(_validation_set1) as feed:
        print('Cr = {0}'.format(lr.calc_cost(feed)))

    yr, x = 18, [1, 98]
    y = lr.predict(x)
    print('{0}= {1}, args: {3}, real: {2}'.format(lr, y, yr, x))

    print('----------- example 2 ------------')
    lr = LinearRegression(2, 250, 0.0000001, 50)
    with CorrelatedIterator(_training_set2) as feed:
        lr.batch_train(feed)
        print('Ct = {0}'.format(lr.calc_cost(feed)))
    with CorrelatedIterator(_validation_set2) as feed:
        print('Cr = {0}'.format(lr.calc_cost(feed)))

    yr, x = 358, [1, 1814, 4]
    y = lr.predict(x)
    print('{0}= {1}, args: {3}, real: {2}'.format(lr, y, yr, x))
