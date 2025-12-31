"""Reduction operations with automatic differentiation.

This module implements reduction operations: Sum, Mean.

Author: TinyAI Team
Version: 0.1.0
"""

from typing import List, Optional
from tinytorch.tensor import Tensor
from tinytorch.autograd.function import Function


class Sum(Function):
    """Sum reduction: y = sum(x, axis, keepdims)
    
    Forward: y = sum(x, axis, keepdims)
    Backward: dL/dx = broadcast(dL/dy) to x.shape
    """
    
    def __init__(self, axis: Optional[int] = None, keepdims: bool = False):
        """Initialize sum reduction.
        
        Args:
            axis: Axis to sum over (None for all)
            keepdims: Whether to keep reduced dimensions
        """
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for sum."""
        self.input_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for sum."""
        # Broadcast gradient back to input shape
        if self.keepdims:
            # Shape already matches after broadcasting
            grad_x = grad_output._broadcast_to(self.input_shape)
        else:
            # Need to add back the reduced dimension
            if self.axis is None:
                # Summed over all dimensions
                grad_x = Tensor([grad_output.data[0]] * self.input_shape.size, 
                              self.input_shape, grad_output.dtype)
            else:
                # Summed over specific axis
                # Expand dimension and broadcast
                new_shape_list = list(grad_output.shape.dims)
                axis = self.axis if self.axis >= 0 else self.input_shape.ndim + self.axis
                new_shape_list.insert(axis, 1)
                grad_expanded = grad_output.reshape(tuple(new_shape_list))
                grad_x = grad_expanded._broadcast_to(self.input_shape)
        
        return [grad_x]


class Mean(Function):
    """Mean reduction: y = mean(x, axis, keepdims)
    
    Forward: y = mean(x, axis, keepdims)
    Backward: dL/dx = broadcast(dL/dy / count) to x.shape
    """
    
    def __init__(self, axis: Optional[int] = None, keepdims: bool = False):
        """Initialize mean reduction.
        
        Args:
            axis: Axis to average over (None for all)
            keepdims: Whether to keep reduced dimensions
        """
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for mean."""
        self.input_shape = x.shape
        
        # Compute count for backward
        if self.axis is None:
            self.count = x.shape.size
        else:
            axis = self.axis if self.axis >= 0 else x.shape.ndim + self.axis
            self.count = x.shape.dims[axis]
        
        return x.mean(axis=self.axis, keepdims=self.keepdims)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward pass for mean."""
        # Divide gradient by count
        grad_output = grad_output.div(self.count)
        
        # Broadcast gradient back to input shape (same as Sum)
        if self.keepdims:
            grad_x = grad_output._broadcast_to(self.input_shape)
        else:
            if self.axis is None:
                grad_x = Tensor([grad_output.data[0]] * self.input_shape.size,
                              self.input_shape, grad_output.dtype)
            else:
                new_shape_list = list(grad_output.shape.dims)
                axis = self.axis if self.axis >= 0 else self.input_shape.ndim + self.axis
                new_shape_list.insert(axis, 1)
                grad_expanded = grad_output.reshape(tuple(new_shape_list))
                grad_x = grad_expanded._broadcast_to(self.input_shape)
        
        return [grad_x]
