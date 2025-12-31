"""Neural network module - Building blocks for neural networks.

This module provides layers, modules, and utilities for building neural networks.

Classes:
    Module: Base class for all neural network modules
    Parameter: Trainable parameter class
"""

from tinytorch.nn.module import Module
from tinytorch.nn.parameter import Parameter
from tinytorch.nn.container import Sequential, ModuleList
from tinytorch.nn import init
from tinytorch.nn import layers

# 导入常用层
from tinytorch.nn.layers import Linear, ReLU, Sigmoid, Tanh, LeakyReLU
from tinytorch.nn.layers import LayerNorm, Dropout, Embedding

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
    'init',
    'layers',
]
