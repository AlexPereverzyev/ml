
TrainingSet1 = """
        test_data/linear.csv|@N,one,X,Y|:6"""
ValidationSet1 = """
        test_data/linear.csv|@N,one,X,Y|7:"""

TrainingSet2 = """
        test_data/logistic.csv|@N,one,X1,X2,Y|:9"""
ValidationSet2 = """
        test_data/logistic.csv|@N,one,X1,X2,Y|10:"""

TrainingSet3 = """
        data/CLR-Garbage Collection-GC Time Spent.csv|@date,one,value|:1000;
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
        data/CLR-Garbage Collection-GC Time Spent.csv|@date,one,value|3000:;
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

TrainingSet5 = """
        test_data/logistic2.csv|X1,X2,Y|:80"""
ValidationSet5 = """
        test_data/logistic2.csv|X1,X2,Y|81:"""

TrainingSet6 = """
        test_data/random_words.csv|sell,car,date,money,free,discount,open,weekend,deal,market,orange,apple,Y|:10"""
ValidationSet6 = """
        test_data/random_words.csv|sell,car,date,money,free,discount,open,weekend,deal,market,orange,apple,Y|11:"""
