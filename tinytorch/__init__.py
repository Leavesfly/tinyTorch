"""tinyTorch - 一个用纯 Python 实现的轻量级深度学习框架。

tinyTorch 的灵感来自 tinyai-deeplearning 项目和 PyTorch，提供
一个清晰且具有教育意义的核心深度学习功能实现。

作者：TinyAI Team
版本：0.1.0
Python: 3.7+

模块：
    ndarr: 多维数组运算
    autograd: 自动微分引擎
    nn: 神经网络层和模块
    ml: 机器学习训练框架
    utils: 工具函数

示例：
    >>> from tinytorch import NdArray, Tensor
    >>> from tinytorch.nn import Linear, Sequential
    >>> from tinytorch.ml import Model, Trainer
"""

import logging as _logging

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version('tinytorch')
except Exception:
    __version__ = '0.1.0'

__author__ = 'TinyAI Team'

# 遵循 Python 库的 logging 最佳实践：为顶层 logger 挂一个 NullHandler，
# 在用户未配置日志时静默，避免 "No handlers could be found" 警告。
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# 导入核心模块
from tinytorch import ndarr
from tinytorch import autograd
from tinytorch import nn
from tinytorch import ml
from tinytorch import utils
from tinytorch import constants

# 方便导入常用类
from tinytorch.ndarr import NdArray, Shape
from tinytorch.autograd import Tensor, Function, no_grad

__all__ = [
    'ndarr',
    'autograd',
    'nn',
    'ml',
    'utils',
    'constants',
    '__version__',
    '__author__',
    'NdArray',
    'Shape',
    'Tensor',
    'Function',
    'no_grad',
]
