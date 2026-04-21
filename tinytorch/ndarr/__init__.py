"""多维数组模块。

本模块提供 tinyTorch 框架中用于数值计算的基础多维数组数据结构和运算。

类:
    NdArray: 多维数组类
    Shape: 形状管理类
"""

from tinytorch.ndarr.ndarray import NdArray
from tinytorch.ndarr.shape import Shape

__all__ = ['NdArray', 'Shape']
