"""Matrix and tensor transformation operations with automatic differentiation.

This module implements matrix operations: MatMul, Transpose, Reshape.

Author: TinyAI Team
Version: 0.1.0
"""

from typing import List, Tuple, Union
from tinytorch.tensor import Tensor
from tinytorch.autograd.function import Function


class MatMul(Function):
    """Matrix multiplication: C = A @ B
    
    Forward: C = A @ B
    Backward: dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
    """
    
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Forward pass for matrix multiplication."""
        self.save_for_backward(a, b)
        return a.matmul(b)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for matrix multiplication."""
        a, b = self.get_saved_tensors()
        
        # dL/dA = dL/dC @ B^T
        grad_a = grad_output.matmul(b.transpose())
        
        # dL/dB = A^T @ dL/dC
        grad_b = a.transpose().matmul(grad_output)
        
        return [grad_a, grad_b]


class Transpose(Function):
    """Transpose operation: B = A^T
    
    Forward: B = transpose(A, axes)
    Backward: dL/dA = transpose(dL/dB, reverse_axes)
    """
    
    def __init__(self, axes: Tuple[int, ...] = None):
        """Initialize transpose operation.
        
        Args:
            axes: Permutation of dimensions (None for default reverse)
        """
        super().__init__()
        self.axes = axes
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for transpose."""
        return x.transpose(self.axes)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for transpose."""
        if self.axes is None:
            # Default transpose: reverse all dimensions
            grad_x = grad_output.transpose()
        else:
            # Invert the permutation
            inv_axes = [0] * len(self.axes)
            for i, ax in enumerate(self.axes):
                inv_axes[ax] = i
            grad_x = grad_output.transpose(tuple(inv_axes))
        
        return [grad_x]


class Reshape(Function):
    """Reshape operation: y = reshape(x, new_shape)
    
    Forward: y = reshape(x, new_shape)
    Backward: dL/dx = reshape(dL/dy, old_shape)
    """
    
    def __init__(self, new_shape: Union[Tuple[int, ...], List[int]]):
        """Initialize reshape operation.
        
        Args:
            new_shape: Target shape
        """
        super().__init__()
        self.new_shape = new_shape
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for reshape."""
        self.old_shape = x.shape
        return x.reshape(self.new_shape)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for reshape."""
        return [grad_output.reshape(self.old_shape.dims)]
