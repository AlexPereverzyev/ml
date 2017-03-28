
from data_tools import CorrelatedIterator
from predictor import LinearRegression

if __name__ == "__main__":
    data_feed = CorrelatedIterator('test_data/*.csv|Date,Value')

    # test
    for data in (d for i, d in enumerate(data_feed) if i < 10):
        print (data)

    # lr = LinearRegression()
    # lr.batch_train(data_feed)
    # y = lr.predict([1, 1, 1, 1])
    # print('{0}= {1}'.format(lr, y))
