
class CsvIterator(object):
    """CSV file line-by-line iterator, accepts list of columns and
    returns one tuple a time from CSV file ensuring specified
    columns oreder"""

    separator = ','

    def __init__(self, file_name, columns):
        self.file_name = file_name.strip()
        if not self.file_name:
            raise Exception('Invalid file name')
        self.columns = ([col for col in
                        (c.strip().lower() for c in columns) if col]
                        if columns else [])
        self.pos = 0

    def __iter__(self):
        self.data_file = open(self.file_name, 'r', encoding='utf-8')
        columns = [c.strip().lower() for c in
                   self.data_file.readline().split(self.separator)]
        self.pos = 1
        if not columns:
            raise StopIteration('CSV header not found')
        self.indexes = [i for i in
                        (columns.index(c) for c in
                         self.columns if c in columns)]
        if not self.indexes and not self.columns:
            self.indexes = range(len(columns))
        if not self.indexes:
            raise StopIteration('Specified columns not found')
        return self

    def __next__(self):
        line = self.data_file.readline()
        self.pos += 1
        if not line:
            self.data_file.close()
            raise StopIteration('EOF reached')
        data = line.split(self.separator)
        select = [data[i].strip() for i in self.indexes]
        return select

    def close(self):
        self.pos = 0
        if self.data_file:
            self.data_file.close()
