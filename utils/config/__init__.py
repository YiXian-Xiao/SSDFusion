import toml
import argparse
import copy


__all__ = [
    'Config',
    'ConfigLoader',
    'adapters'
]


class Config(argparse.Namespace):
    def __init__(self):
        super().__init__()

    def to_dict(self):
        result = {}

        for key, value in vars(self).items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], Config):
                values = []
                for idx, value in enumerate(value):
                    values.append(value.to_dict())
            else:
                result[key] = value

        return result

    def state_dict(self, parent=None, path=None, result=None):
        if result is None:
            result = {}
            path = ''

        for key, value in vars(self).items():
            if key.startswith('__'):
                continue

            if isinstance(value, Config):
                value.state_dict(self, path + f'{key}.', result)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], Config):
                for idx, value in enumerate(value):
                    value.state_dict(self, path + f'{key}.' + idx, result)
            else:
                result[path + key] = value

        return result

    def __getitem__(self, item):
        if isinstance(item, tuple) or isinstance(item, list):
            if not hasattr(self, item[0]):
                return item[1]
            else:
                return getattr(self, item[0])

        try:
            return getattr(self, item)
        except AttributeError:
            pass

        return None

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, item):
        return hasattr(self, item)

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)


class ConfigLoader(argparse.Namespace):

    def __init__(self, parent=None, **kwargs):
        self.__parent__ = parent
        super().__init__(**kwargs)

    def load_from_dict(self, data: dict, root=None):
        if root is None:
            root = self

        for key in data:
            value = data[key]

            if isinstance(value, dict):
                if ':ref' in value:
                    ref = copy.deepcopy(self.get_root().resolve_value(value[':ref']))
                    setattr(self, key, ref)

                    if isinstance(ref, ConfigLoader):
                        del(value[':ref'])
                        ref.merge_with_dict(value)
                else:
                    subcfg = ConfigLoader(self)
                    setattr(self, key, subcfg)
                    subcfg.load_from_dict(value, root)

            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                lst = []
                setattr(self, key, lst)

                for entry in value:
                    subcfg = ConfigLoader(self)
                    subcfg.load_from_dict(entry, root)
                    lst.append(subcfg)

            else:
                setattr(self, key, value)

    def merge_with_dict(self, data: dict, root=None):
        if root is None:
            root = self

        for key in data:
            value = data[key]

            if isinstance(value, dict):
                if ':ref' in value:
                    ref = copy.deepcopy(self.get_root().resolve_value(value[':ref']))
                    setattr(self, key, ref)
                    ref.merge_with_dict(value)
                elif hasattr(self, key) and isinstance(getattr(self, key), ConfigLoader):
                    subcfg = getattr(self, key)
                    subcfg.merge_with_dict(value, root)
                else:
                    subcfg = ConfigLoader(self)
                    setattr(self, key, subcfg)
                    subcfg.load_from_dict(value, root)

            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                lst = []
                setattr(self, key, lst)
                for entry in value:
                    subcfg = ConfigLoader(self)
                    subcfg.load_from_dict(entry, root)
                    lst.append(subcfg)

            else:
                setattr(self, key, value)

    def merge_with_config(self, cfg):
        for key in vars(cfg):
            value = getattr(cfg, key)

            if isinstance(value, ConfigLoader) and hasattr(self, key):
                subcfg = getattr(self, key)
                subcfg.merge_with_config(value)

            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                lst = []
                setattr(self, key, lst)
                for entry in value:
                    subcfg = ConfigLoader(self)
                    subcfg.load_from_dict(entry)
                    lst.append(subcfg)

            else:
                setattr(self, key, value)

    def load_from_path(self, path):
        self.load_from_dict(toml.load(path))

    @classmethod
    def load_from_file(cls, path):
        cfg = ConfigLoader()
        cfg.load_from_path(path)

        return cfg

    def resolve_value(self, path: str):
        ref = self
        for name in path.split('.'):
            if hasattr(ref, name):
                ref = getattr(ref, name)
            else:
                return

        return ref

    def get_root(self):
        root = self
        while root.__parent__ is not None:
            root = root.__parent__

        return root

    def state_dict(self, parent=None, path=None, result=None):
        if result is None:
            result = {}
            path = ''

        for key, value in vars(self).items():
            if key.startswith('__'):
                continue

            if isinstance(value, ConfigLoader):
                value.state_dict(self, path + f'{key}.', result)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], ConfigLoader):
                for idx, value in enumerate(value):
                    value.state_dict(self, path + f'{key}.' + idx, result)
            else:
                result[path + key] = value

        return result

    def to_dict(self):
        result = {}

        for key, value in vars(self).items():
            if key.startswith('__'):
                continue

            if isinstance(value, ConfigLoader):
                result[key] = value.to_dict()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], ConfigLoader):
                values = []
                for idx, value in enumerate(value):
                    values.append(value.to_dict())
            else:
                result[key] = value

        return result

    def save_to(self, path):
        data = self.to_dict()
        toml.dump(data, path)

    def clone(self):
        return copy.deepcopy(self)

    def freeze(self):
        result = Config()

        for key, value in vars(self).items():
            if key.startswith('__'):
                continue

            if isinstance(value, ConfigLoader):
                result[key] = value.freeze()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], ConfigLoader):
                values = []
                for idx, value in enumerate(value):
                    values.append(value.freeze())
            else:
                result[key] = value

        return result

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            cfg = ConfigLoader()
            cfg.load_from_dict(value, self.__root__)
            value = cfg

        super().__setattr__(key, value)

    def __getitem__(self, item):
        if isinstance(item, tuple) or isinstance(item, list):
            if not hasattr(self, item[0]):
                return item[1]
            else:
                return getattr(self, item[0])

        try:
            return getattr(self, item)
        except AttributeError:
            pass

        return None

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, item):
        return hasattr(self, item)

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)
