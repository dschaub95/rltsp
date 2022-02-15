import json


class Settings(dict):
    """Experiment configuration options.

    Wrapper around in-built dict class to access members through the dot operation.

    Experiment parameters:
    """

    def __init__(self, config_dict):
        super(Settings, self).__init__()
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
    """Returns default config object."""
    return Settings(json.load(open("./configs/default.json")))


def get_config(filepath=None):
    """Returns config from json file."""
    config = get_default_config()
    if filepath is not None:
        config.update(Settings(json.load(open(filepath))))
    return config


import pandas as pd
import json
import inspect
import yaml

# base config class
# import all the modules which shall be represented in


class Config:
    def __init__(
        self,
        config_dict=None,
        config_class=None,
        config_yaml=None,
        config_json=None,
        restrictive=True,
    ) -> None:

        # set default values as specified inside the config class
        self._defaults()
        if config_dict is not None:
            self.from_dict(config_dict, restrictive)
        elif config_class is not None:
            self.from_class(config_class, restrictive)
        elif config_yaml is not None:
            self.from_yaml(config_yaml, restrictive)
        elif config_json is not None:
            self.from_json(config_json, restrictive)

    def _defaults(self):
        # specifiy all internal default values of this config
        # or get them froman external config file which is predetermined
        pass

    def set_defaults(self, _class, restrictive=False):
        # retrieve defaults form another class
        full_args_spec = inspect.getfullargspec(_class.__init__)
        args = [arg for arg in full_args_spec.args if not arg == "self"]
        defaults = full_args_spec.defaults
        if defaults is None:
            pass
        else:
            defaults = list(defaults)
            args = args[::-1][: len(defaults)][::-1]
            defaults_dict = dict(zip(args, defaults))
            self.from_dict(defaults_dict, restrictive=restrictive)

    # if needed add overwrite feature
    def from_dict(self, config_dict, restrictive=True):
        # only set keys that are in both dicts
        if restrictive:
            reference_keys = [key for key in self.__dict__ if key in config_dict]
        else:
            reference_keys = config_dict.keys()
        for key in reference_keys:
            # how to handle nested input dict
            # create new subconfigs based on structure
            # call from dict recursively for subconfigs
            if type(config_dict[key]) == dict:
                self.__dict__[key] = Config()
                self.__dict__[key].from_dict(config_dict[key], restrictive)
            # write values for all keys
            self.__dict__[key] = config_dict[key]

    def from_class(self, config_class, restrictive=True):
        self.from_dict(config_class.__dict__, restrictive)

    def from_yaml(self, config_yaml, restrictive=True):
        # read via pyaml
        with open(config_yaml, "r") as f:
            self.from_dict(yaml.safe_load(f), restrictive)

    def from_json(self, config_json, restrictive=True):
        # read via json
        with open(config_json, "r") as f:
            self.from_dict(json.load(f), restrictive)

    def to_dict(self, nested=True):
        # should be recursive in case we want to have sub configs for seperate modules
        # and write these to file or anything else
        if nested:
            nested_dict = dict()
            for key in self.__dict__:
                try:
                    sub_dict = self.__dict__[key].to_dict(nested=True)
                    nested_dict[key] = sub_dict
                except:
                    nested_dict[key] = self.__dict__[key]
            return nested_dict
        else:
            return self.__dict__

    def to_df(self, nested=True):
        return pd.DataFrame.from_dict(self.to_dict(nested), orient="index").transpose()

    def to_yaml(self, file_name, nested=True):
        with open(file_name, "w") as outfile:
            yaml.dump(self.to_dict(nested=nested), outfile, default_flow_style=False)

    def _flatten_dict(self, nested_dict):
        flattend_dict = dict()
        for key in nested_dict:
            if type(nested_dict[key]) == dict:
                # if sub dict is nested again flatten it
                flat_sub_dict = self._flatten_dict(nested_dict[key])
                for sub_key in flat_sub_dict:
                    key_phrase = "_".join(key.split("_")[0:-1])
                    new_key = f"{key_phrase}.{sub_key}"
                    flattend_dict[new_key] = flat_sub_dict[sub_key]
            else:
                flattend_dict[key] = nested_dict[key]
        return flattend_dict

    def to_flattend_dict(self):
        # flatten nested dict
        nested_dict = self.to_dict(nested=True)
        return self._flatten_dict(nested_dict)

    def __repr__(self):
        return str(self.to_dict(nested=True))
