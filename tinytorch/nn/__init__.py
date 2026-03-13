"""神经网络模块 - 神经网络的构建块。

该模块提供构建神经网络所需的层、模块和工具。

类：
    Module: 所有神经网络模块的基类
    Parameter: 可训练参数类
"""

from tinytorch.nn.module import Module
from tinytorch.nn.parameter import Parameter
from tinytorch.nn.container import Sequential, ModuleList
from tinytorch.nn import init
from tinytorch.nn import layers

# 导入常用层
from tinytorch.nn.layers import Linear, ReLU, Sigmoid, Tanh, LeakyReLU
from tinytorch.nn.layers import LayerNorm, Dropout, Embedding
from tinytorch.nn.layers import Conv2d, RNN, LSTM, GRU, MultiHeadAttention

__all__ = [
    'Module',
    'Parameter',
    'Sequential',
    'ModuleList',
    'Linear',
    'ReLU',
    'Sigmoid',
    'Tanh',
    'LeakyReLU',
    'LayerNorm',
    'Dropout',
    'Embedding',
    'Conv2d',
    'RNN',
    'LSTM',
    'GRU',
    'MultiHeadAttention',
    'init',
    'layers',
]
