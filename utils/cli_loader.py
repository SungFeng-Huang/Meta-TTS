"""
Code from: https://github.com/omni-us/jsonargparse/issues/117
keyword:
    value
    - effect
__base__:
    str or list[str],(each str should be a relative path from cur cofig file)
    - Merge every config one by one, current last.
__delete__:
    True or str|int or list[str|int],True for delete all keys from other config,
    str|int only delete the specific key (for dict) or index (for list)
    - Delete some part of config from other.
__import__:
    Any
    - Just delete this, for convenience of reference in yaml
change_item:
    list[[index, item]],used only when merge list
    - Add ability of merg list, change the list[index] from other to item
insert_item:
    list[[index, item, (extend)]],used only when merge list
    - Add ability of merg list, insert iterm to the list at index, extend=True if insert a list of items
pre_item:
    Anyor list[Any],used only when merge list
    - Add ability of merg list, add the value in the start of the list from other to item
post_item:
    Anyor list[Any],used only when merge list
    - Add ability of merg list, add the value in the end of the list from other to item

"""
import os
import re
import copy
from typing import Any, Dict, List

import yaml
from jsonargparse import Path, get_config_read_mode, set_loader
from jsonargparse.loaders_dumpers import yaml_load
from jsonargparse.util import change_to_path_dir
from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI

def deep_update(source, override):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    if isinstance(source, Dict) and isinstance(override, Dict):
        if '__delete__' in override:
            delete_keys = override.pop('__delete__')
            if isinstance(delete_keys, str):
                delete_keys = [delete_keys]

            if isinstance(delete_keys, list):
                for k in delete_keys:
                    if k in source:
                        source.pop(k)
            elif delete_keys:
                return override
        for key, value in override.items():
            if isinstance(value, Dict) and key in source:
                source[key] = deep_update(source[key], value)
            else:
                source[key] = override[key]
        return source
    elif isinstance(source, List) and isinstance(override, Dict):
        if '__delete__' in override and override['__delete__'] is True:
            override.pop('__delete__')
            return override

        if 'change_item' in override:
            change_item = override.pop('change_item')
            for index, v in change_item:
                source[index] = deep_update(source[index], v)

        if 'insert_item' in override:
            insert_item = override.pop('insert_item')
            insert_item.sort(key = lambda x: x[0], reverse = True)
            for item in insert_item:
                if len(item) == 3:
                    index, value, extend = item
                else:
                    index, value = item
                    extend = False
                if extend:
                    assert isinstance(value, list), 'Cannot extend a non-list'
                    value.reverse()
                    for v in value:
                        source.insert(index, v)
                else:
                    source.insert(index, value)

                if '__delete__' in override:
                    if isinstance(override['__delete__'], int):
                        override['__delete__'] = [override['__delete__']]
                    for i in range(len(override['__delete__'])):
                        if override['__delete__'][i] >= index:
                            if extend:
                                override['__delete__'][i] += len(value)
                            else:
                                override['__delete__'][i] += 1

        if '__delete__' in override:
            delete_keys = override.pop('__delete__')
            if isinstance(delete_keys, int):
                delete_keys = [delete_keys]

            if isinstance(delete_keys, list):
                delete_keys = list({int(d) for d in delete_keys})
                delete_keys.sort(reverse = True)
                for k in delete_keys:
                    source.pop(k)
            elif delete_keys:
                return override
        if 'pre_item' in override:
            source = (override['pre_item'] if isinstance(override['pre_item'], list) else [override['pre_item']]) + source
        if 'post_item' in override:
            source = source + (override['post_item'] if isinstance(override['post_item'], list) else [override['post_item']])
        return source
    return override


def get_cfg_from_path(cfg_path):
    fpath = Path(cfg_path, mode = get_config_read_mode())
    with change_to_path_dir(fpath):
        cfg_str = fpath.get_content()
        parsed_cfg = yaml_load(cfg_str)
    return parsed_cfg


def parse_config(cfg_file, cfg_path = None, **kwargs):
    if '__base__' in cfg_file:
        sub_cfg_paths = cfg_file.pop('__base__')
        if sub_cfg_paths is not None:
            if not isinstance(sub_cfg_paths, list):
                sub_cfg_paths = [sub_cfg_paths]
            sub_cfg_paths = [sub_cfg_path if isinstance(sub_cfg_path, list) else [sub_cfg_path, ''] for sub_cfg_path in sub_cfg_paths]
            if cfg_path is not None:
                sub_cfg_paths = [[os.path.normpath(os.path.join(os.path.dirname(cfg_path), sub_cfg_path[0])) if not os.path.isabs(
                    sub_cfg_path[0]) else sub_cfg_path[0], sub_cfg_path[1]] for sub_cfg_path in sub_cfg_paths]
            sub_cfg_file = {}
            for sub_cfg_path in sub_cfg_paths:
                cur_cfg_file = parse_path(sub_cfg_path[0], **kwargs)
                for key in sub_cfg_path[1].split('.'):
                    if key:
                        cur_cfg_file = cur_cfg_file[key]
                sub_cfg_file = deep_update(sub_cfg_file, cur_cfg_file)
            cfg_file = deep_update(sub_cfg_file, cfg_file)
    if '__import__' in cfg_file:
        cfg_file.pop('__import__')

    for k, v in cfg_file.items():
        if isinstance(v, dict):
            cfg_file[k] = parse_config(v, cfg_path, **kwargs)
    return cfg_file


def parse_path(cfg_path, seen_cfg = None, **kwargs):
    abs_cfg_path = os.path.abspath(cfg_path)
    if seen_cfg is None:
        seen_cfg = {}
    elif abs_cfg_path in seen_cfg:
        if seen_cfg[abs_cfg_path] is None:
            raise RuntimeError('Circular reference detected in config file')
        else:
            return copy.deepcopy(seen_cfg[abs_cfg_path])

    cfg_file = get_cfg_from_path(cfg_path)
    seen_cfg[abs_cfg_path] = None
    cfg_file = parse_config(cfg_file, cfg_path = cfg_path, seen_cfg = seen_cfg, **kwargs)
    seen_cfg[abs_cfg_path] = cfg_file
    return cfg_file


def parse_str(cfg_str, cfg_path = None, seen_cfg = None, **kwargs):
    if seen_cfg is None:
        seen_cfg = {}
    cfg_file = yaml_load(cfg_str)
    if cfg_path is not None:
        abs_cfg_path = os.path.abspath(cfg_path)
        if abs_cfg_path in seen_cfg:
            if seen_cfg[abs_cfg_path] is None:
                raise RuntimeError('Circular reference detected in config file')
            else:
                return copy.deepcopy(seen_cfg[abs_cfg_path])
        seen_cfg[abs_cfg_path] = None
    if isinstance(cfg_file, dict):
        cfg_file = parse_config(cfg_file, cfg_path = cfg_path, seen_cfg = seen_cfg, **kwargs)
    if cfg_path is not None:
        seen_cfg[abs_cfg_path] = cfg_file
    return cfg_file


def yaml_with_merge_load(stream, path = None, ext_vars = None):
    config = parse_str(stream, path = path)
    if ext_vars is not None and isinstance(ext_vars, dict) and isinstance(config, dict):
        config = deep_update(config, ext_vars)
    return config


set_loader('yaml_with_merge', yaml_with_merge_load)


class ArgumentParser(LightningArgumentParser):
    def __init__(self, parser_mode: str = 'yaml_with_merge', *args: Any, **kwargs: Any) -> None:
        super().__init__(parser_mode = parser_mode, *args, **kwargs)

class CLI(LightningCLI):
    def init_parser(self, **kwargs: Any) -> LightningArgumentParser:
        """Method that instantiates the argument parser."""
        return ArgumentParser(**kwargs)
