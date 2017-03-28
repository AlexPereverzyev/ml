
from functools import reduce
from data_tools.csv_iter import CsvIterator
from data_tools.data_expression import DataExpression


class CorrelatedIterator(object):
    """Data feeder which reads data from files specified by expression
    and outputs one training sample per time. The samples can be joined
    by specified column"""

    def __init__(self, expression):
        self.expression = expression

    def __iter__(self):
        self.iterators = [iter(CsvIterator(data_file, columns))
                          for data_file, columns in
                          DataExpression(self.expression)]
        if not self.iterators:
            raise Exception('Invalid iteration expression')
        return self

    def __next__(self):
        chunks = []
        for itr in self.iterators:
            data = next(itr)
            chunks.append(data)
        return reduce(lambda x, y: x + y, chunks)
