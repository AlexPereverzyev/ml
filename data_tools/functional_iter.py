
from data_tools.csv_iter import CsvIterator


class FunctionalIterator(CsvIterator):
    """Wraps CSV iterator to add calculated and constant columns"""

    def __next__(self):
        data = super().__next__()
        if not self.columns:
            return data
        result = []
        i = 0
        for c in self.columns:
            if c.is_indexed:
                d = data[i]
                try:
                    d = float(d)
                except Exception:
                    pass
                if c.is_func:
                    d = c.func(d)
                result.append(d)
                i += 1
            elif c.is_const:
                result.append(c.const)
        return result
