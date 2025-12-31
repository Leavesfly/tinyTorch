"""Variable class for automatic differentiation.

This module provides the Variable class which wraps Tensor and enables
automatic gradient computation through dynamic computational graphs.

Author: TinyAI Team
Version: 0.1.0
"""

from typing import Optional
from tinytorch.tensor import Tensor, Shape


class Variable:
    """Automatic differentiation variable.
    
    Variable wraps a Tensor and maintains gradient information and
    computational graph connections for automatic differentiation.
    
    Attributes:
        value: The tensor value
        grad: Gradient tensor (same shape as value)
        creator: Function that created this variable
        requires_grad: Whether to compute gradients
        name: Variable name for debugging
    
    Example:
        >>> x = Variable(Tensor([[1.0, 2.0]]), name="x")
        >>> y = x * 2 + 1
        >>> y.backward()
        >>> print(x.grad)
    """
    
    def __init__(self, value: Tensor, name: str = None, requires_grad: bool = True):
        """Initialize a Variable.
        
        Args:
            value: Tensor value
            name: Variable name (optional)
            requires_grad: Whether to track gradients
        """
        if not isinstance(value, Tensor):
            raise TypeError(f"Variable value must be Tensor, got {type(value)}")
        
        self.value = value
        self.grad = None
        self.creator = None
        self.requires_grad = requires_grad
        self.name = name if name else f"var_{id(self)}"
    
    def backward(self, retain_graph: bool = False):
        """Compute gradients via backpropagation.
        
        This implements reverse-mode automatic differentiation using
        topological sorting of the computational graph.
        
        Args:
            retain_graph: Whether to keep the computational graph
        """
        if not self.requires_grad:
            return
        
        # Initialize gradient as ones (for scalar output)
        if self.grad is None:
            self.grad = Tensor.ones(self.value.shape, self.value.dtype)
        
        # Topological sort to get execution order
        topo_order = []
        visited = set()
        
        def build_topo(var):
            if var not in visited and var.creator is not None:
                visited.add(var)
                for input_var in var.creator.inputs:
                    build_topo(input_var)
                topo_order.append(var)
        
        build_topo(self)
        
        # Backward pass in reverse topological order
        for var in reversed(topo_order):
            if var.creator is None:
                continue
            
            # Get gradient with respect to this variable's output
            grad_output = var.grad
            
            # Compute gradients for inputs
            grad_inputs = var.creator.backward(grad_output)
            
            # Accumulate gradients to input variables
            for input_var, grad_input in zip(var.creator.inputs, grad_inputs):
                if input_var.requires_grad:
                    if input_var.grad is None:
                        input_var.grad = grad_input
                    else:
                        # Accumulate gradients (for variables used multiple times)
                        input_var.grad = input_var.grad.add(grad_input)
        
        # Clean up graph if not retaining
        if not retain_graph:
            self.unchain_backward()
    
    def unchain_backward(self):
        """Release computational graph to free memory."""
        def clear_graph(var):
            if var.creator is not None:
                for input_var in var.creator.inputs:
                    clear_graph(input_var)
                var.creator = None
        
        clear_graph(self)
    
    def clear_grad(self):
        """Clear gradient."""
        self.grad = None
    
    def detach(self) -> 'Variable':
        """Create a new Variable detached from computation graph.
        
        Returns:
            New Variable with same value but no gradient tracking
        """
        return Variable(self.value.copy(), self.name + "_detached", requires_grad=False)
    
    # ==================== Arithmetic Operations ====================
    
    def add(self, other) -> 'Variable':
        """Addition operation."""
        from tinytorch.autograd.ops.basic import Add
        
        if isinstance(other, (int, float)):
            other = Variable(Tensor([other]), requires_grad=False)
        
        return Add()(self, other)
    
    def sub(self, other) -> 'Variable':
        """Subtraction operation."""
        from tinytorch.autograd.ops.basic import Sub
        
        if isinstance(other, (int, float)):
            other = Variable(Tensor([other]), requires_grad=False)
        
        return Sub()(self, other)
    
    def mul(self, other) -> 'Variable':
        """Multiplication operation."""
        from tinytorch.autograd.ops.basic import Mul
        
        if isinstance(other, (int, float)):
            other = Variable(Tensor([other]), requires_grad=False)
        
        return Mul()(self, other)
    
    def div(self, other) -> 'Variable':
        """Division operation."""
        from tinytorch.autograd.ops.basic import Div
        
        if isinstance(other, (int, float)):
            other = Variable(Tensor([other]), requires_grad=False)
        
        return Div()(self, other)
    
    def neg(self) -> 'Variable':
        """Negation operation."""
        from tinytorch.autograd.ops.basic import Neg
        return Neg()(self)
    
    def pow(self, exponent: float) -> 'Variable':
        """Power operation."""
        from tinytorch.autograd.ops.math_ops import Pow
        return Pow(exponent)(self)
    
    # ==================== Math Operations ====================
    
    def exp(self) -> 'Variable':
        """Exponential operation."""
        from tinytorch.autograd.ops.math_ops import Exp
        return Exp()(self)
    
    def log(self) -> 'Variable':
        """Natural logarithm operation."""
        from tinytorch.autograd.ops.math_ops import Log
        return Log()(self)
    
    def sqrt(self) -> 'Variable':
        """Square root operation."""
        from tinytorch.autograd.ops.math_ops import Sqrt
        return Sqrt()(self)
    
    # ==================== Matrix Operations ====================
    
    def matmul(self, other: 'Variable') -> 'Variable':
        """Matrix multiplication."""
        from tinytorch.autograd.ops.matrix import MatMul
        return MatMul()(self, other)
    
    def transpose(self, axes=None) -> 'Variable':
        """Transpose operation."""
        from tinytorch.autograd.ops.matrix import Transpose
        return Transpose(axes)(self)
    
    def reshape(self, new_shape) -> 'Variable':
        """Reshape operation."""
        from tinytorch.autograd.ops.matrix import Reshape
        return Reshape(new_shape)(self)
    
    def sum(self, axis=None, keepdims=False) -> 'Variable':
        """Sum reduction."""
        from tinytorch.autograd.ops.reduce import Sum
        return Sum(axis, keepdims)(self)
    
    def mean(self, axis=None, keepdims=False) -> 'Variable':
        """Mean reduction."""
        from tinytorch.autograd.ops.reduce import Mean
        return Mean(axis, keepdims)(self)
    
    # ==================== Activation Functions ====================
    
    def relu(self) -> 'Variable':
        """ReLU activation."""
        from tinytorch.autograd.ops.activation import ReLU
        return ReLU()(self)
    
    def sigmoid(self) -> 'Variable':
        """Sigmoid activation."""
        from tinytorch.autograd.ops.activation import Sigmoid
        return Sigmoid()(self)
    
    def tanh(self) -> 'Variable':
        """Tanh activation."""
        from tinytorch.autograd.ops.activation import Tanh
        return Tanh()(self)
    
    # ==================== Properties ====================
    
    @property
    def shape(self) -> Shape:
        """Get tensor shape."""
        return self.value.shape
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return self.value.shape.ndim
    
    @property
    def size(self) -> int:
        """Get total number of elements."""
        return self.value.shape.size
    
    @property
    def dtype(self) -> str:
        """Get data type."""
        return self.value.dtype
    
    # ==================== Operator Overloading ====================
    
    def __add__(self, other):
        """Addition operator."""
        return self.add(other)
    
    def __radd__(self, other):
        """Right addition operator."""
        return self.add(other)
    
    def __sub__(self, other):
        """Subtraction operator."""
        return self.sub(other)
    
    def __rsub__(self, other):
        """Right subtraction operator."""
        if isinstance(other, (int, float)):
            other = Variable(Tensor([other]), requires_grad=False)
        return other.sub(self)
    
    def __mul__(self, other):
        """Multiplication operator."""
        return self.mul(other)
    
    def __rmul__(self, other):
        """Right multiplication operator."""
        return self.mul(other)
    
    def __truediv__(self, other):
        """Division operator."""
        return self.div(other)
    
    def __rtruediv__(self, other):
        """Right division operator."""
        if isinstance(other, (int, float)):
            other = Variable(Tensor([other]), requires_grad=False)
        return other.div(self)
    
    def __neg__(self):
        """Negation operator."""
        return self.neg()
    
    def __pow__(self, exponent):
        """Power operator."""
        return self.pow(exponent)
    
    def __matmul__(self, other):
        """Matrix multiplication operator."""
        return self.matmul(other)
    
    def __repr__(self) -> str:
        """String representation."""
        grad_str = f", grad={self.grad}" if self.grad is not None else ""
        return f"Variable(name={self.name}, shape={self.value.shape}, requires_grad={self.requires_grad}{grad_str})"
    
    def __str__(self) -> str:
        """String representation."""
        return self.__repr__()
