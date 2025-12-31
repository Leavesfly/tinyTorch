"""Activation function operations with automatic differentiation.

This module implements activation functions: ReLU, Sigmoid, Tanh.

Author: TinyAI Team
Version: 0.1.0
"""

import math
from typing import List
from tinytorch.tensor import Tensor
from tinytorch.autograd.function import Function


class ReLU(Function):
    """ReLU activation: y = max(0, x)
    
    Forward: y = max(0, x)
    Backward: dL/dx = dL/dy * (x > 0)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for ReLU."""
        self.save_for_backward(x)
        return x.relu()
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for ReLU."""
        x, = self.get_saved_tensors()
        
        # Gradient is 1 where x > 0, else 0
        mask_data = [1.0 if val > 0 else 0.0 for val in x.data]
        mask = Tensor(mask_data, x.shape, x.dtype)
        
        grad_x = grad_output.mul(mask)
        return [grad_x]


class Sigmoid(Function):
    """Sigmoid activation: y = 1 / (1 + exp(-x))
    
    Forward: y = sigmoid(x)
    Backward: dL/dx = dL/dy * y * (1 - y)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for sigmoid."""
        y = x.sigmoid()
        self.save_for_backward(y)
        return y
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for sigmoid."""
        y, = self.get_saved_tensors()
        
        # dy/dx = y * (1 - y)
        one = Tensor.ones(y.shape, y.dtype)
        grad_x = grad_output.mul(y).mul(one.sub(y))
        return [grad_x]


class Tanh(Function):
    """Tanh activation: y = tanh(x)
    
    Forward: y = tanh(x)
    Backward: dL/dx = dL/dy * (1 - y^2)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for tanh."""
        y = x.tanh()
        self.save_for_backward(y)
        return y
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for tanh."""
        y, = self.get_saved_tensors()
        
        # dy/dx = 1 - y^2
        one = Tensor.ones(y.shape, y.dtype)
        grad_x = grad_output.mul(one.sub(y.pow(2)))
        return [grad_x]
