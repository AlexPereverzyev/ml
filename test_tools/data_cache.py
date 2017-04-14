from data_tools.data_iter import DataIterator


class DictionaryCache(dict):
    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        with DataIterator(key) as data_generator:
            data = list(data_generator)
        self[key] = data
        return data
