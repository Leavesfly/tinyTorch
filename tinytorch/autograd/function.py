"""Function base class for automatic differentiation operations.

This module provides the Function base class that all differentiable operations
must inherit from. It defines the interface for forward and backward propagation.

Author: TinyAI Team
Version: 0.1.0
"""

from typing import List, Tuple
from tinytorch.tensor import Tensor


class Function:
    """Base class for all differentiable functions.
    
    Subclasses must implement forward() and backward() methods.
    The Function maintains references to input variables and saved tensors
    for backward propagation.
    
    Attributes:
        inputs: List of input Variable objects
        outputs: List of output Variable objects
        saved_tensors: List of tensors saved during forward pass
    
    Example:
        >>> class Add(Function):
        ...     def forward(self, x, y):
        ...         return x.add(y)
        ...     def backward(self, grad_output):
        ...         return [grad_output, grad_output]
    """
    
    def __init__(self):
        """Initialize Function."""
        self.inputs = []
        self.outputs = []
        self.saved_tensors = []
    
    def __call__(self, *inputs):
        """Call the function (same as call method).
        
        Args:
            *inputs: Variable inputs
            
        Returns:
            Output Variable(s)
        """
        return self.call(*inputs)
    
    def call(self, *inputs):
        """Execute forward propagation and build computation graph.
        
        This method:
        1. Extracts tensor values from input Variables
        2. Calls forward() to compute output tensors
        3. Wraps output tensors in Variables
        4. Sets up backward graph connections
        
        Args:
            *inputs: Variable inputs
            
        Returns:
            Output Variable (or tuple of Variables for multi-output)
        """
        # Import here to avoid circular dependency
        from tinytorch.autograd.variable import Variable
        
        # Store input variables
        self.inputs = list(inputs)
        
        # Extract tensor values
        input_tensors = [var.value for var in inputs]
        
        # Forward propagation
        output_tensors = self.forward(*input_tensors)
        
        # Handle single or multiple outputs
        if isinstance(output_tensors, (list, tuple)):
            # Multiple outputs
            output_vars = []
            for tensor in output_tensors:
                var = Variable(tensor)
                var.creator = self
                output_vars.append(var)
            self.outputs = output_vars
            return tuple(output_vars)
        else:
            # Single output
            output_var = Variable(output_tensors)
            output_var.creator = self
            self.outputs = [output_var]
            return output_var
    
    def forward(self, *inputs: Tensor) -> Tensor:
        """Forward propagation computation.
        
        Subclasses must implement this method to define the forward computation.
        
        Args:
            *inputs: Input tensors
            
        Returns:
            Output tensor(s)
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Backward propagation computation.
        
        Subclasses must implement this method to compute input gradients
        given the output gradient.
        
        Args:
            grad_output: Gradient of loss with respect to output
            
        Returns:
            List of gradients with respect to each input
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement backward()")
    
    def save_for_backward(self, *tensors: Tensor):
        """Save tensors needed for backward computation.
        
        Args:
            *tensors: Tensors to save
        """
        self.saved_tensors = list(tensors)
    
    def get_saved_tensors(self) -> Tuple[Tensor, ...]:
        """Retrieve saved tensors.
        
        Returns:
            Tuple of saved tensors
        """
        return tuple(self.saved_tensors)
