"""Basic arithmetic operations with automatic differentiation.

This module implements basic arithmetic operations: Add, Sub, Mul, Div, Neg.

Author: TinyAI Team
Version: 0.1.0
"""

from typing import List
from tinytorch.tensor import Tensor
from tinytorch.autograd.function import Function


class Add(Function):
    """Element-wise addition: z = x + y
    
    Forward: z = x + y
    Backward: dL/dx = dL/dz, dL/dy = dL/dz
    """
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass for addition."""
        return x.add(y)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for addition.
        
        Gradients flow equally to both inputs.
        Handle broadcasting by summing over broadcast dimensions.
        """
        x_shape = self.inputs[0].value.shape
        y_shape = self.inputs[1].value.shape
        
        grad_x = grad_output
        grad_y = grad_output
        
        # Handle broadcasting: sum over broadcast dimensions
        if grad_x.shape != x_shape:
            grad_x = self._sum_to_shape(grad_x, x_shape)
        
        if grad_y.shape != y_shape:
            grad_y = self._sum_to_shape(grad_y, y_shape)
        
        return [grad_x, grad_y]
    
    @staticmethod
    def _sum_to_shape(tensor: Tensor, target_shape) -> Tensor:
        """Sum tensor to target shape (handle broadcasting in backward)."""
        # Sum over extra dimensions
        ndim_diff = tensor.shape.ndim - target_shape.ndim
        for _ in range(ndim_diff):
            tensor = tensor.sum(axis=0, keepdims=False)
        
        # Sum over dimensions where target is 1
        for i in range(target_shape.ndim):
            if target_shape[i] == 1 and tensor.shape[i] > 1:
                tensor = tensor.sum(axis=i, keepdims=True)
        
        return tensor


class Sub(Function):
    """Element-wise subtraction: z = x - y
    
    Forward: z = x - y
    Backward: dL/dx = dL/dz, dL/dy = -dL/dz
    """
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass for subtraction."""
        return x.sub(y)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for subtraction."""
        x_shape = self.inputs[0].value.shape
        y_shape = self.inputs[1].value.shape
        
        grad_x = grad_output
        grad_y = grad_output.neg()
        
        # Handle broadcasting
        if grad_x.shape != x_shape:
            grad_x = Add._sum_to_shape(grad_x, x_shape)
        
        if grad_y.shape != y_shape:
            grad_y = Add._sum_to_shape(grad_y, y_shape)
        
        return [grad_x, grad_y]


class Mul(Function):
    """Element-wise multiplication: z = x * y
    
    Forward: z = x * y
    Backward: dL/dx = dL/dz * y, dL/dy = dL/dz * x
    """
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass for multiplication."""
        self.save_for_backward(x, y)
        return x.mul(y)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for multiplication."""
        x, y = self.get_saved_tensors()
        x_shape = self.inputs[0].value.shape
        y_shape = self.inputs[1].value.shape
        
        grad_x = grad_output.mul(y)
        grad_y = grad_output.mul(x)
        
        # Handle broadcasting
        if grad_x.shape != x_shape:
            grad_x = Add._sum_to_shape(grad_x, x_shape)
        
        if grad_y.shape != y_shape:
            grad_y = Add._sum_to_shape(grad_y, y_shape)
        
        return [grad_x, grad_y]


class Div(Function):
    """Element-wise division: z = x / y
    
    Forward: z = x / y
    Backward: dL/dx = dL/dz / y, dL/dy = -dL/dz * x / y^2
    """
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass for division."""
        self.save_for_backward(x, y)
        return x.div(y)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for division."""
        x, y = self.get_saved_tensors()
        x_shape = self.inputs[0].value.shape
        y_shape = self.inputs[1].value.shape
        
        grad_x = grad_output.div(y)
        grad_y = grad_output.neg().mul(x).div(y.pow(2))
        
        # Handle broadcasting
        if grad_x.shape != x_shape:
            grad_x = Add._sum_to_shape(grad_x, x_shape)
        
        if grad_y.shape != y_shape:
            grad_y = Add._sum_to_shape(grad_y, y_shape)
        
        return [grad_x, grad_y]


class Neg(Function):
    """Negation: y = -x
    
    Forward: y = -x
    Backward: dL/dx = -dL/dy
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for negation."""
        return x.neg()
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for negation."""
        return [grad_output.neg()]
