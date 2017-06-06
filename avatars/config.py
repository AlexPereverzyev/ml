
import yaml
import logging
import logging.config


__all__ = ['current']

_config_path = 'config.yaml'


class AppConfig(object):
    def __init__(self, entries):
        for k, v in entries.items():
            attr = k.replace('-', '_')
            self.__dict__[attr] = AppConfig(v) if type(v) is dict else v


with open(_config_path, 'r') as stream:
    current = AppConfig(yaml.load(stream))

with open(current.logging_config_path, 'r') as stream:
    logging_config = yaml.load(stream)
    logging.config.dictConfig(logging_config)
