from pathlib import Path

import yaml
from easydict import EasyDict


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('\n%s.%s = edict()' % (pre, key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)
        except:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value


def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            """
                yaml.load(): 将 YAML 格式的字符串或文件内容转换成 Python 对象
                如果 YAML 数据表示了一个键值对的集合,yaml.load() 可能会返回一个 Python 字典，其中 YAML 中的键值对被映射为字典中的键值对
            """
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


cfg = EasyDict()
"""
    Path(__file__): __file__ 是一个内建变量,用于获取当前脚本(或模块)的文件路径。Path(__file__) 创建一个 Path 对象，表示当前脚本的文件路径

        * .resolve(): resolve() 方法用于获取路径的绝对路径，即将路径转换为标准的绝对路径形式。
        * .parent: parent 属性用于获取路径的父目录，即当前文件的上一级目录。
        * / '../': 使用 / 运算符将当前路径与相对路径 '../' 拼接起来。这是通过 pathlib 的特性，它允许使用 / 运算符来连接路径部分。
        * .resolve(): 再次调用 resolve() 方法，将拼接后的路径转换为其绝对路径
"""
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0
