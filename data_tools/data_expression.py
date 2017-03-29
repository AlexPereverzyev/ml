
import re
import os
import sys
from collections import namedtuple


class DataExpression(object):
    """Data expression parser with iterator interface. Expression syntax:
    expression := source [ | column [ | range ] ];
        source := path [ file_name.ext | *.ext ]
        column := col1 [ , col2 , ... ]
         range := bound : bound | bound : | : bound
         bound := 1 .. N

    Returns tuples with the following structure:
    ( file_name, (columns list, correlation column, left bound, right bound) )
    """

    separator = ','
    source_separator = ';'
    columns_separator = '|'
    range_separator = ':'
    column_mark = '^'

    single_file_pattern = re.compile('.+?([^/*.]+\.[\w]+)$')
    masked_file_pattern = re.compile('(.+)/\*(\.[\w]+)$')
    directory_pattern = re.compile('/?[^/*.]+/?$')

    Settings = namedtuple('Settings',
                          ['columns', 'correlation', 'left', 'right'])

    def __init__(self, expression):
        if not expression:
            raise Exception('Missing data expression')
        self.expression = expression

    def __iter__(self):
        self.sources = self.parse_sources()
        return self

    def __next__(self):
        return next(self.sources)

    def parse_sources(self):
        for source_expr in (src for src in
                            (s.strip() for s in
                             self.expression.split(self.source_separator))
                            if src):
            source, columns_expr, range_expr = self._split(source_expr)
            columns, marked = self._parse_columns(columns_expr)
            left, right = self._parse_range(range_expr)
            is_single = self.single_file_pattern.match(source)
            is_masked = self.masked_file_pattern.match(source)
            is_folder = self.directory_pattern.match(source)
            if is_single:
                yield (source,
                       self.Settings(columns, marked, left, right))
            elif is_masked:
                dir = is_masked.group(1)
                ext = is_masked.group(2).lower()
                for f in os.listdir(dir):
                    if f.lower().endswith(ext):
                        file_name = os.path.join(dir, f)
                        yield (file_name,
                               self.Settings(columns, marked, left, right))
            elif is_folder:
                for f in os.listdir(source):
                    file_name = os.path.join(source, f)
                    yield (file_name,
                           self.Settings(columns, marked, left, right))

    def _split(self, source_expr):
        split = source_expr.split(self.columns_separator)
        if not split or len(split) > 3:
            raise Exception('Invalid source expression')
        source = split[0].strip()
        if not source:
            raise Exception('Invalid source expression')
        columns_expr = split[1] if len(split) > 1 else None
        range_expr = split[2] if len(split) > 2 else None
        if (columns_expr and self.range_separator in columns_expr and
           (not range_expr or self.range_separator not in range_expr)):
            columns_expr, range_expr = range_expr, columns_expr
        return source, columns_expr, range_expr

    def _parse_columns(self, columns_expr):
        if not columns_expr:
            return None, None
        columns = [c.strip() for c in columns_expr.split(self.separator) if c]
        marked = next((c for c in columns if c.endswith(self.column_mark) and
                       c.replace(self.column_mark, '')), None)
        columns = [col for col in
                   (c.replace(self.column_mark, '') for c in columns) if col]
        marked = marked.replace(self.column_mark, '') if marked else None
        return columns, marked

    def _parse_range(self, range_expr):
        if not range_expr:
            return 1, sys.maxsize
        if self.range_separator not in range_expr:
            raise Exception('Invalid range expression')
        try:
            left, right = range_expr.split(self.range_separator)
            left, right = left.strip(), right.strip()
            left, right = (int(left) if left else 1,
                           int(right) if right else sys.maxsize)
        except Exception as e:
            raise Exception('Invalid range expression', e)
        return left, right
