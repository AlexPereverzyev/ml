
from data_tools import DataIterator
from sklearn import linear_model
from functools import reduce

TrainingSet1 = """
        test_data/linear.csv|@N,one,X,Y|:6"""

ValidationSet1 = """
        test_data/linear.csv|@N,one,X,Y|7:"""

TrainingSet2 = """
        test_data/logistic.csv|@N,one,X1,X2,Y|:9"""

ValidationSet2 = """
        test_data/logistic.csv|@N,one,X1,X2,Y|10:"""

TrainingSet3 = """
        data/CLR-Garbage Collection-GC Time Spent.csv|@date,one,value|:500;
        data/CLR-Locks and Threads-Current Physical Threads.csv|@date,value;
        data/Hardware Resources-CPU-Busy.csv|@date,value;
        data/Hardware Resources-Memory-Used.csv|@date,value;
        data/Hardware Resources-Disks-Writes.csv|@date,value;
        data/Hardware Resources-Disks-Reads.csv|@date,value;
        data/Hardware Resources-Network-Outgoing packet.csv|@date,value;
        data/Hardware Resources-Network-Incoming packets.csv|@date,value;
        data/IIS-CPU.csv|@date,value;
        data/Calls Per Minute.csv|@date,value;
        data/Errors Per Minute.csv|@date,value;
        data/Average Response Time.csv|@date,value>400;
    """

ValidationSet3 = """
        data/CLR-Garbage Collection-GC Time Spent.csv|@date,one,value|500:1000;
        data/CLR-Locks and Threads-Current Physical Threads.csv|@date,value;
        data/Hardware Resources-CPU-Busy.csv|@date,value;
        data/Hardware Resources-Memory-Used.csv|@date,value;
        data/Hardware Resources-Disks-Writes.csv|@date,value;
        data/Hardware Resources-Disks-Reads.csv|@date,value;
        data/Hardware Resources-Network-Outgoing packet.csv|@date,value;
        data/Hardware Resources-Network-Incoming packets.csv|@date,value;
        data/IIS-CPU.csv|@date,value;
        data/Calls Per Minute.csv|@date,value;
        data/Errors Per Minute.csv|@date,value;
        data/Average Response Time.csv|@date,value>400;
    """

TrainingSet4 = """
        test_data/logistic2.csv|one,X1,X2,Y|:50"""

ValidationSet4 = """
        test_data/logistic2.csv|one,X1,X2,Y|51:"""


def test_regression(name, features, iterations, learning, precision,
                    training_set, validation_set, details_count,
                    algorithm):
    print('-' * 20, name, '-' * 20)
    with DataIterator(training_set) as data_generator:
        feed = list(data_generator)
    regression = algorithm(features, iterations, learning, precision)
    regression.batch_train(feed)
    print(regression)
    print('Et = {0} %'.format(regression.calc_training_error(feed)))
    with DataIterator(validation_set) as data_generator:
        feed = list(data_generator)
    print('E  = {0} %'.format(regression.calc_training_error(feed)))
    for d in (d for i, d in enumerate(feed) if i < details_count):
        print('Ye = {0}, Yr = {1}'.format(regression.predict(d), d[-1]))


def test_regression_skl(name, precision, training_set, validation_set, details_count,
                       algorithm):
    print('-' * 20, name, '-' * 20)
    with DataIterator(training_set) as data_generator:
        feed = list(data_generator)
    X = [f[1:-1] for f in feed]
    Y = [f[-1] for f in feed]
    regression = algorithm()
    regression.fit(X, Y)
    print('Y = ', regression.coef_)
    print('Et = ?')
    # print('Et = {0} %'.format(regression.calc_training_error(feed)))
    with DataIterator(validation_set) as data_generator:
        feed = list(data_generator)
    # todo: refactor
    missed_count = list(map(lambda d:
                        int(abs(regression.predict(d[1:-1])[0] - d[-1]) >  precision),
                        feed))
    error = reduce(lambda x, y: x + y, missed_count) / len(missed_count)
    error *= 100
    print('E  = {0} %'.format(error))

    for d in (d for i, d in enumerate(feed) if i < details_count):
        x = d[1:-1]
        print('Ye = {0}, Yr = {1}'.format(regression.predict(x)[0], d[-1]))
