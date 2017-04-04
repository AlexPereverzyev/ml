
import re
import os
import sys
from collections import namedtuple
from enum import Enum


class DataExpression(object):
    """Data expression parser with iterator interface. Expression syntax:
    expression := source [ | column [ | range ] ];
        source := path [ file.ext | *.ext ]
        column := col [ , col , ... ]
           col := name | @name | name^power | const
         const := one | zero
         range := bound : bound | bound : | : bound
         bound := 1 .. N

    Returns tuples with the following structure:
    ( file_name, (columns list, correlation column, left bound, right bound) )
    """

    source_separator = ';'
    syntax_separator = '|'
    separator = ','
    join_mark = '@'
    power_mark = '^'
    range_separator = ':'

    terminals = [separator, source_separator, syntax_separator,
                 range_separator, join_mark, power_mark]

    const_columns = {'zero': 0, 'one': 1}

    single_file_pattern = re.compile('.+?([^/*.]+\.[\w]+)$')
    masked_file_pattern = re.compile('(.+)/\*(\.[\w]+)$')
    directory_pattern = re.compile('/?[^/*.]+/?$')

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
        tokens = self._tokenize()
        if not tokens:
            raise Exception('Missing data expression')
        source, i = None, -1
        state = ParserState.Source
        while True:
            i, token = self._next_token(i, tokens)
            if not token:
                if source:
                    state = ParserState.Done
                else:
                    break
            if state == ParserState.Source:
                source = [s for s in self._parse_source(token)]
                columns, join = [], None
                left, right = None, None
                state = ParserState.Columns
            elif state == ParserState.Columns:
                if token == self.source_separator:
                    i -= 1
                    state = ParserState.Done
                elif token == self.syntax_separator:
                    if columns:
                        state = ParserState.Range
                elif token == self.separator:
                    pass
                elif token == self.join_mark:
                    i += 1
                    join = tokens[i]
                    columns.append(ColumnMeta(join))
                elif token == self.power_mark:
                    i += 1
                    token = tokens[i]
                    power = int(token)
                    columns[-1].func = lambda x: x ** power
                else:
                    columns.append(ColumnMeta(token))
                    if token in self.const_columns.keys():
                        columns[-1].const = self.const_columns[token]
            elif state == ParserState.Range:
                if token == self.source_separator:
                    i -= 1
                    state = ParserState.Done
                elif token == self.syntax_separator:
                    pass
                elif token == self.range_separator:
                    left = left or 1
                else:
                    if not left:
                        left = int(token)
                    else:
                        right = int(token)
            elif state == ParserState.Done:
                for s in source:
                    yield (s, SourceMeta(
                        columns, join, left or 1, right or sys.maxsize))
                source = None
                state = ParserState.Source

    def _tokenize(self):
        t = ''
        tokens = []
        for c in self.expression:
            if c in self.terminals:
                t = t.strip().lower()
                if t:
                    tokens.append(t)
                t = ''
                tokens.append(c)
            else:
                t += c
        t = t.strip().lower()
        if t:
            tokens.append(t)
        return tokens

    def _parse_source(self, source):
        is_single = self.single_file_pattern.match(source)
        is_masked = self.masked_file_pattern.match(source)
        is_folder = self.directory_pattern.match(source)
        if is_single:
            yield source
        elif is_masked:
            dir = is_masked.group(1)
            ext = is_masked.group(2)
            for f in os.listdir(dir):
                if f.lower().endswith(ext):
                    file_name = os.path.join(dir, f).lower()
                    yield file_name
        elif is_folder:
            for f in os.listdir(source):
                file_name = os.path.join(source, f).lower()
                yield file_name

    def _next_token(self, i, tokens):
        if i < len(tokens) - 1:
            i += 1
            token = tokens[i]
            return i, token
        return i, None


SourceMeta = namedtuple('SourceMeta', ['columns', 'join', 'left', 'right'])


class ColumnMeta(str):
    func = None
    const = None
    index = None

    @property
    def is_func(self):
        return self.func is not None

    @property
    def is_const(self):
        return self.const is not None

    @property
    def is_indexed(self):
        return self.index is not None


class ParserState(Enum):
    Source = 0
    Columns = 1
    Range = 2
    Done = 3
