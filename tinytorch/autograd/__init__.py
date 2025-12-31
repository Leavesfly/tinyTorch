"""Autograd module - Automatic differentiation engine.

This module implements automatic differentiation for building dynamic
computational graphs and computing gradients.

Classes:
    Variable: Automatic differentiation variable
    Function: Function base class for operations
"""

from tinytorch.autograd.variable import Variable
from tinytorch.autograd.function import Function

__all__ = ['Variable', 'Function']
