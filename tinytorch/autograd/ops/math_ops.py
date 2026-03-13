"""Mathematical function operations with automatic differentiation.

This module implements mathematical functions: Exp, Log, Sqrt, Pow.

Author: TinyAI Team
Version: 0.1.0
"""

from typing import List
from tinytorch.ndarr import NdArray
from tinytorch.autograd.function import Function


class Exp(Function):
    """Exponential function: y = exp(x)
    
    Forward: y = exp(x)
    Backward: dL/dx = dL/dy * exp(x) = dL/dy * y
    """
    
    def forward(self, x: NdArray) -> NdArray:
        """Forward pass for exp."""
        y = x.exp()
        self.save_for_backward(y)
        return y
    
    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """Backward pass for exp."""
        y, = self.get_saved_tensors()
        return [grad_output.mul(y)]


class Log(Function):
    """Natural logarithm: y = log(x)
    
    Forward: y = log(x)
    Backward: dL/dx = dL/dy / x
    """
    
    def __init__(self, epsilon: float = 1e-10):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x: NdArray) -> NdArray:
        """Forward pass for log."""
        self.save_for_backward(x)
        return x.log()
    
    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """Backward pass for log. Clamps x to avoid division by zero."""
        x, = self.get_saved_tensors()
        x_safe = x.add(self.epsilon)
        return [grad_output.div(x_safe)]


class Sqrt(Function):
    """Square root: y = sqrt(x)
    
    Forward: y = sqrt(x)
    Backward: dL/dx = dL/dy / (2 * sqrt(x)) = dL/dy / (2 * y)
    """
    
    def __init__(self, epsilon: float = 1e-10):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x: NdArray) -> NdArray:
        """Forward pass for sqrt."""
        y = x.sqrt()
        self.save_for_backward(y)
        return y
    
    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """Backward pass for sqrt. Clamps denominator to avoid division by zero at x=0."""
        y, = self.get_saved_tensors()
        denom = y.mul(2.0).add(self.epsilon)
        return [grad_output.div(denom)]


class Pow(Function):
    """Power function: y = x^n
    
    Forward: y = x^n
    Backward: dL/dx = dL/dy * n * x^(n-1)
    """
    
    def __init__(self, exponent: float, epsilon: float = 1e-10):
        """Initialize power function.
        
        Args:
            exponent: Power exponent
            epsilon: Small value to avoid 0^(n-1) when n < 1
        """
        super().__init__()
        self.exponent = exponent
        self.epsilon = epsilon
    
    def forward(self, x: NdArray) -> NdArray:
        """Forward pass for power."""
        self.save_for_backward(x)
        return x.pow(self.exponent)
    
    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """Backward pass for power. Clamps x when exponent < 1 to avoid 0^(negative)."""
        x, = self.get_saved_tensors()
        exp_minus_1 = self.exponent - 1
        x_safe = x.add(self.epsilon) if exp_minus_1 < 0 else x
        grad = grad_output.mul(x_safe.pow(exp_minus_1).mul(self.exponent))
        return [grad]
