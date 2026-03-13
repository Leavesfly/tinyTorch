"""Tensor module - Multi-dimensional array operations.

This module provides the fundamental tensor data structure and operations
for numerical computing in tinyTorch framework.

Classes:
    Tensor: Multi-dimensional array class
    Shape: Shape management class
"""

from tinytorch.tensor.tensor import Tensor
from tinytorch.tensor.shape import Shape

__all__ = ['Tensor', 'Shape']
