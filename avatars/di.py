
import config
import logging

__all__ = ['inject']

_factories = {logging.Logger.__name__:
              lambda t: logging.getLogger(t.__name__)}

_singeltons = {config.AppConfig.__name__:
               config.current}


def inject(ctor):
    """Dependency injection decorator for __init__ methods"""
    def init(*args, **kwargs):
        if hasattr(ctor, '__annotations__'):
            for k, dep_type in ctor.__annotations__.items():
                name = dep_type.__name__
                if name in _singeltons:
                    dep = _singeltons[name]
                elif name in _factories:
                    target_type = type(args[0])
                    dep = _factories[name](target_type)
                else:
                    dep = dep_type()
                kwargs[k] = dep
        ctor(*args, **kwargs)
    return init
