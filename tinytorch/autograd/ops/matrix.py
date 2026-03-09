"""带自动微分的矩阵和张量变换运算。

该模块实现了矩阵运算：MatMul（矩阵乘法）、Transpose（转置）、Reshape（重塑）。

作者：TinyAI Team
版本：0.1.0
"""

from typing import List, Tuple, Union
from tinytorch.tensor import Tensor
from tinytorch.autograd.function import Function


class MatMul(Function):
    """矩阵乘法：C = A @ B
    
    前向传播：C = A @ B
    反向传播：dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
    """
    
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """矩阵乘法的前向传播。"""
        self.save_for_backward(a, b)
        return a.matmul(b)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """矩阵乘法的反向传播。"""
        a, b = self.get_saved_tensors()
        
        # dL/dA = dL/dC @ B^T
        grad_a = grad_output.matmul(b.transpose())
        
        # dL/dB = A^T @ dL/dC
        grad_b = a.transpose().matmul(grad_output)
        
        return [grad_a, grad_b]


class Transpose(Function):
    """转置运算：B = A^T
    
    前向传播：B = transpose(A, axes)
    反向传播：dL/dA = transpose(dL/dB, reverse_axes)
    """
    
    def __init__(self, axes: Tuple[int, ...] = None):
        """初始化转置运算。
        
        Args:
            axes: 维度排列（None 表示默认反转）
        """
        super().__init__()
        self.axes = axes
    
    def forward(self, x: Tensor) -> Tensor:
        """转置的前向传播。"""
        return x.transpose(self.axes)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """转置的反向传播。"""
        if self.axes is None:
            # 默认转置：反转所有维度
            grad_x = grad_output.transpose()
        else:
            # 逆排列
            inv_axes = [0] * len(self.axes)
            for i, ax in enumerate(self.axes):
                inv_axes[ax] = i
            grad_x = grad_output.transpose(tuple(inv_axes))
        
        return [grad_x]


class Reshape(Function):
    """重塑运算：y = reshape(x, new_shape)
    
    前向传播：y = reshape(x, new_shape)
    反向传播：dL/dx = reshape(dL/dy, old_shape)
    """
    
    def __init__(self, new_shape: Union[Tuple[int, ...], List[int]]):
        """初始化重塑运算。
        
        Args:
            new_shape: 目标形状
        """
        super().__init__()
        self.new_shape = new_shape
    
    def forward(self, x: Tensor) -> Tensor:
        """重塑的前向传播。"""
        self.old_shape = x.shape
        return x.reshape(self.new_shape)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """重塑的反向传播。"""
        return [grad_output.reshape(self.old_shape.dims)]
