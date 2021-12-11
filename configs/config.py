import json


class Config(dict):
    """Experiment configuration options.

    Wrapper around in-built dict class to access members through the dot operation.

    Experiment parameters:
    """

    def __init__(self, config_dict):
        super(Config, self).__init__()
        for key in config_dict:
            self[key] = config_dict[key]

    def __getattr__(self, attr):
        return self[attr]

    def __setitem__(self, key, value):
        return super().__setitem__(key, value)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    __delattr__ = dict.__delitem__


def get_default_config():
    """Returns default config object.
    """
    return Config(json.load(open("./configs/default.json")))


def get_config(filepath=None):
    """Returns config from json file.
    """
    config = get_default_config()
    if filepath is not None:
        config.update(Config(json.load(open(filepath))))
    return config
