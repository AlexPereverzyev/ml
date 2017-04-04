
class CsvIterator(object):
    """CSV file line-by-line iterator, accepts list of columns and
    returns one tuple a time from CSV file ensuring specified
    columns oreder"""

    separator = ','

    def __init__(self, file_name, columns):
        self.file_name = file_name
        if not self.file_name:
            raise Exception('Invalid file name')
        self.columns = columns
        self.pos = 0

    def __iter__(self):
        self.data_file = open(self.file_name, 'r', encoding='utf-8')
        self.file_columns = [c.strip().lower() for c in
                             self.data_file.readline().split(self.separator)]
        self.pos = 1
        if not self.file_columns:
            raise StopIteration('CSV header not found')
        for c in self.columns:
            if c in self.file_columns:
                c.index = self.file_columns.index(c)
        if (self.columns and
           not next((c for c in self.columns if c.is_indexed), None)):
            raise StopIteration('Specified columns not found')
        return self

    def __next__(self):
        line = self.data_file.readline()
        self.pos += 1
        if not line:
            self.data_file.close()
            raise StopIteration('EOF reached')
        data = line.split(self.separator)
        select = [data[c.index].strip() for c in self.columns if c.is_indexed]
        select = select or [d.strip() for d in data]
        return select

    def close(self):
        self.pos = 0
        if self.data_file:
            self.data_file.close()
