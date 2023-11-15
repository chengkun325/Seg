import pydoc
import sys
from importlib import import_module
from pathlib import Path
from typing import Union
from addict import Dict


class ConfigDict(Dict):
    """
    继承自 Python 的内置 Dict 类，并重写了两个主要的方法: __missing__ 和 __getattr__。
    """
    def __missing__(self, name):
        """
        如果访问的键不存在，会引发一个 KeyError 的异常。
        :param name:
        :return:
        """
        raise KeyError(name)

    def __getattr__(self, name):
        """
        如果访问的属性值不存在，会首先尝试调用父类（即 Dict 类）的 __getattr__ 方法来获取属性。
        如果父类中不存在该属性，则会引发一个 AttributeError 异常。如果存在，则直接返回该属性的值。
        :param name:
        :return:
        """
        try:
            value = super().__getattr__(name)
        except KeyError:
            # f-string格式化字符串。用{}里的值替换f。
            ex = AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        else:
            return value
        raise ex


def py2dict(file_path: Union[str, Path]) -> dict:
    """
    将 Python 文件转换为一个字典。
    :param file_path:
    :return:
    """
    # 将路径转化为绝对路径。
    file_path = Path(file_path).absolute()
    if file_path.suffix != ".py":
        raise TypeError(f"Only Py file can be parsed, but got {file_path.name} instead.")
    if not file_path.exists():
        raise FileExistsError(f"There is no file at the path {file_path}")
    # 从文件路径中提取模块名，不包括后缀。
    module_name = file_path.stem
    if "." in module_name:
        raise ValueError("Dots are not allowed in config file path.")
    # 获取文件所在的目录路径。
    config_dir = str(file_path.parent)
    # 在 sys.path 中添加文件所在的目录，这样可以确保 Python 能够导入该文件中的模块。
    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    # 上面添加的目录在这里删除。
    sys.path.pop(0)
    # 从模块的 __dict__ 属性中提取所有的顶级名称和值，并排除以 __ 开头的名称
    # （这通常表示私有属性或方法），然后构建一个字典。
    cfg_dict = {name: value for name, value in mod.__dict__.items() if not name.startswith("__")}
    return cfg_dict


def py2cfg(file_path: Union[str, Path]) -> ConfigDict:
    """
    将 Python 文件转换为一个 ConfigDict 对象。
    :param file_path:
    :return:
    """
    cfg_dict = py2dict(file_path)
    return ConfigDict(cfg_dict)


def object_from_dict(d, parent=None, **default_kwargs):
    """
    根据给定的字典数据创建一个对象。
    :param d: 一个字典
    :param parent: 可选的参数，表示父对象
    :param default_kwargs: 可变的关键字参数，表示默认的键值对
    :return:
    """
    # 复制 d 字典的内容到 kwargs。
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)
    if parent is not None:
        # ** 用于字典的解包。
        return getattr(parent, object_type)(**kwargs)
    return pydoc.locate(object_type)(**kwargs)
