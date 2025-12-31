"""tinyTorch - A lightweight deep learning framework implemented in pure Python.

tinyTorch is inspired by tinyai-deeplearning project and PyTorch, providing
a clean and educational implementation of core deep learning functionality.

Author: TinyAI Team
Version: 0.1.0
Python: 3.7+

Modules:
    tensor: Multi-dimensional array operations
    autograd: Automatic differentiation engine
    nn: Neural network layers and modules
    ml: Machine learning training framework
    utils: Utility functions

Example:
    >>> from tinytorch import Tensor, Variable
    >>> from tinytorch.nn import Linear, Sequential
    >>> from tinytorch.ml import Model, Trainer
"""

__version__ = '0.1.0'
__author__ = 'TinyAI Team'

# Import core modules
from tinytorch import tensor
from tinytorch import autograd
from tinytorch import nn
from tinytorch import ml
from tinytorch import utils

# Import commonly used classes for convenience
from tinytorch.tensor import Tensor, Shape
from tinytorch.autograd import Variable, Function

__all__ = [
    'tensor',
    'autograd',
    'nn',
    'ml',
    'utils',
    '__version__',
    '__author__',
    'Tensor',
    'Shape',
    'Variable',
    'Function',
]
