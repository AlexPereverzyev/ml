from functools import reduce
from collections import OrderedDict, namedtuple
from data_tools.csv_iter import CsvIterator
from data_tools.data_expression import DataExpression


class CorrelatedIterator(object):
    """Data feeder which reads data from files specified by expression
    and outputs one data sample per time. The samples can be joined by
    specified column"""

    def __init__(self, expression):
        self.expression = expression
        if not self.expression:
            raise Exception('Invalid data expression')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self):
        self.iter_state = OrderedDict(
            (f, SourceState(s, iter(CsvIterator(f, s.columns))))
            for f, s in DataExpression(self.expression))
        if not self.iter_state:
            raise Exception('Invalid data expression')
        for f, ss in self.iter_state.items():
            while ss.iterator.pos < ss.meta.left:
                next(ss.iterator)
        return self

    def __next__(self):
        data = []
        for f, ss in self.iter_state.items():
            while True:
                if ss.iterator.pos > ss.meta.right:
                    self.close()
                    raise StopIteration('Exceeded right bound')
                try:
                    d = next(ss.iterator)
                except StopIteration:
                    self.close()
                    raise
                for i, v in enumerate(d):
                    try:
                        d[i] = float(v)
                    except Exception:
                        continue
                ci = (ss.meta.columns.index(ss.meta.join)
                      if ss.meta.join else None)
                cv = (d[ci]
                      if ss.meta.join else None)
                cs = CorrelatedState(cv, d)
                if not cs.key:
                    break
                del cs.values[ci]
                prev_cs = next((c for c in reversed(data) if c.key), None)
                if not prev_cs or cs.key == prev_cs.key:
                    break
            data.append(cs)
        return reduce(lambda d1, d2: d1 + d2, (s.values for s in data))

    def close(self):
        if self.iter_state:
            for iterator in (s.iterator for s in self.iter_state.values()):
                iterator.close()
            self.iter_state.clear()


SourceState = namedtuple('SourceState', ['meta', 'iterator'])

CorrelatedState = namedtuple('CorrelatedState', ['key', 'values'])
