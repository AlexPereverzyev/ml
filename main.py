
from data_tools import CorrelatedIterator
from predictor import LinearRegression

if __name__ == "__main__":
    feed = CorrelatedIterator(
     """test_data/Average Response Time.csv|date^,value|3:;
        test_data/Calls Per Minute.csv|date^,value;
        test_data/Errors Per Minute.csv|date^,value;""")

    # test
    for data in (d for i, d in enumerate(feed) if i < 15):
        print (data)

    # lr = LinearRegression()
    # lr.batch_train(feed)
    # y = lr.predict([1, 1, 1, 1])
    # print('{0}= {1}'.format(lr, y))
