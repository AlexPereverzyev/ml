
import re
import os


class DataExpression(object):
    """Data expression parser with iterator interface"""

    separator = ','
    source_separator = ';'
    columns_separator = '|'

    single_file_pattern = re.compile('.+?([^/*.]+\.[\w]+)$')
    masked_file_pattern = re.compile('(.+)/\*(\.[\w]+)$')
    directory_pattern = re.compile('/?[^/*.]+/?$')

    def __init__(self, expression):
        if not expression:
            raise Exception('Missing data expression')
        self.expression = expression

    def __iter__(self):
        self.sources = self.parsed_sources()
        return self

    def __next__(self):
        return next(self.sources)

    def parsed_sources(self):
        for source in (src for src in
                       (s.strip() for s in
                        self.expression.split(self.source_separator))
                       if src):
            split = source.split(self.columns_separator)
            if len(split) != 2:  # source expr | columns expr
                raise Exception('Source or columns are not specified')
            # parse columns expression
            columns = [c for c in split[1].split(self.separator) if c]
            if not columns:
                raise Exception('Invalid columns expression')
            # parse source expression
            source_expr = split[0].strip()
            if not source_expr:
                raise Exception('Invalid source expression')
            is_single = self.single_file_pattern.match(source_expr)
            is_masked = self.masked_file_pattern.match(source_expr)
            is_folder = self.directory_pattern.match(source_expr)
            if is_single:
                yield (source_expr, columns)
            elif is_masked:
                dir = is_masked.group(1)
                ext = is_masked.group(2).lower()
                for f in os.listdir(dir):
                    if f.lower().endswith(ext):
                        file_name = os.path.join(dir, f)
                        yield (file_name, columns)
            elif is_folder:
                for f in os.listdir(source_expr):
                    yield (f, columns)
