from functools import reduce
from collections import OrderedDict, namedtuple
from data_tools.csv_iter import CsvIterator
from data_tools.data_expression import DataExpression


class CorrelatedIterator(object):
    """Data feeder which reads data from files specified by expression
    and outputs one data sample per time. The samples can be joined by
    specified column"""

    SourceState = namedtuple('SourceState', ['settings', 'iterator'])
    CorrelatedState = namedtuple('CorrelatedState', ['key', 'values'])

    def __init__(self, expression):
        self.expression = expression
        if not self.expression:
            raise Exception('Invalid data expression')

    def __iter__(self):
        self.iter_state = OrderedDict(
            (f, self.SourceState(s, iter(CsvIterator(f, s.columns))))
            for f, s in DataExpression(self.expression))
        if not self.iter_state:
            raise Exception('Invalid data expression')
        for f, ss in self.iter_state.items():
            while ss.iterator.pos < ss.settings.left:
                next(ss.iterator)
        return self

    def __next__(self):
        data = []
        for f, ss in self.iter_state.items():
            while True:
                if ss.iterator.pos > ss.settings.right:
                    self.close()
                    raise StopIteration('Exceeded right bound')
                try:
                    d = next(ss.iterator)
                except StopIteration:
                    self.close()
                    raise
                ci = (ss.settings.columns.index(ss.settings.correlation)
                      if ss.settings.correlation else None)
                cv = (d[ci]
                      if ss.settings.correlation else None)
                cs = self.CorrelatedState(cv, d)
                if not cs.key:
                    break
                prev_cs = next((c for c in reversed(data) if c.key), None)
                if not prev_cs:
                    break
                if cs.key == prev_cs.key:
                    del cs.values[ci]
                    break
            data.append(cs)
        return reduce(lambda d1, d2: d1 + d2, (s.values for s in data))

    def close(self):
        if self.iter_state:
            for iterator in (s.iterator for s in self.iter_state.values()):
                iterator.close()
            self.iter_state.clear()
