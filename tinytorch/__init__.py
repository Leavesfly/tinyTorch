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

__version__ = '0.1.0'
__author__ = 'TinyAI Team'

# 导入核心模块
from tinytorch import ndarr
from tinytorch import autograd
from tinytorch import nn
from tinytorch import ml
from tinytorch import utils

# 方便导入常用类
from tinytorch.ndarr import NdArray, Shape
from tinytorch.autograd import Tensor, Function, no_grad

__all__ = [
    'ndarr',
    'autograd',
    'nn',
    'ml',
    'utils',
    '__version__',
    '__author__',
    'NdArray',
    'Shape',
    'Tensor',
    'Function',
    'no_grad',
]
