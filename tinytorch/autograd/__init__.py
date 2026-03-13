"""Autograd module - Automatic differentiation engine.

This module implements automatic differentiation for building dynamic
computational graphs and computing gradients.

Classes:
    Tensor: Automatic differentiation variable
    Function: Function base class for operations
"""

from tinytorch.autograd.tensor import Tensor
from tinytorch.autograd.function import Function

__all__ = ['Tensor', 'Function']
