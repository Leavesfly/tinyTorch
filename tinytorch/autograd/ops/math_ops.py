"""Mathematical function operations with automatic differentiation.

This module implements mathematical functions: Exp, Log, Sqrt, Pow.

Author: TinyAI Team
Version: 0.1.0
"""

from typing import List
from tinytorch.tensor import Tensor
from tinytorch.autograd.function import Function


class Exp(Function):
    """Exponential function: y = exp(x)
    
    Forward: y = exp(x)
    Backward: dL/dx = dL/dy * exp(x) = dL/dy * y
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for exp."""
        y = x.exp()
        self.save_for_backward(y)
        return y
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for exp."""
        y, = self.get_saved_tensors()
        return [grad_output.mul(y)]


class Log(Function):
    """Natural logarithm: y = log(x)
    
    Forward: y = log(x)
    Backward: dL/dx = dL/dy / x
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for log."""
        self.save_for_backward(x)
        return x.log()
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for log."""
        x, = self.get_saved_tensors()
        return [grad_output.div(x)]


class Sqrt(Function):
    """Square root: y = sqrt(x)
    
    Forward: y = sqrt(x)
    Backward: dL/dx = dL/dy / (2 * sqrt(x)) = dL/dy / (2 * y)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for sqrt."""
        y = x.sqrt()
        self.save_for_backward(y)
        return y
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for sqrt."""
        y, = self.get_saved_tensors()
        return [grad_output.div(y.mul(2.0))]


class Pow(Function):
    """Power function: y = x^n
    
    Forward: y = x^n
    Backward: dL/dx = dL/dy * n * x^(n-1)
    """
    
    def __init__(self, exponent: float):
        """Initialize power function.
        
        Args:
            exponent: Power exponent
        """
        super().__init__()
        self.exponent = exponent
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for power."""
        self.save_for_backward(x)
        return x.pow(self.exponent)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for power."""
        x, = self.get_saved_tensors()
        grad = grad_output.mul(x.pow(self.exponent - 1).mul(self.exponent))
        return [grad]
