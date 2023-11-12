import json
import yaml
from collections import OrderedDict


def read_yaml(fname):
    with fname.open('rt') as handle:
        class OrderedLoader(yaml.SafeLoader):
            pass
        def construct_mapping(loader, node):
            loader.flatten_mapping(node)
            return OrderedDict(loader.construct_pairs(node))
        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping)
        return yaml.load(handle, OrderedLoader)


def write_yaml(content, fname):
    with fname.open('wt') as handle:
        class OrderedDumper(yaml.SafeDumper):
            pass
        def _dict_representer(dumper, data):
            return dumper.represent_mapping(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                data.items())
        OrderedDumper.add_representer(OrderedDict, _dict_representer)
        return yaml.dump(content, handle, OrderedDumper)


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_config(fname):
    if fname.suffix == '.json':
        return read_json(fname)
    elif fname.suffix == '.yaml':
        return read_yaml(fname)
    else:
        raise NotImplementedError
