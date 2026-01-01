"""自动微分的变量类。

本模块提供了 Variable 类，它包装了 Tensor 并通过动态计算图实现自动梯度计算。

Author: TinyAI Team
Version: 0.1.0
"""

from typing import Optional
from tinytorch.tensor import Tensor, Shape


class Variable:
    """自动微分变量。
    
    Variable 包装了一个 Tensor，并为自动微分维护梯度信息和计算图连接。
    
    属性:
        value: 张量值
        grad: 梯度张量（与 value 形状相同）
        creator: 创建此变量的函数
        requires_grad: 是否计算梯度
        name: 变量名称，用于调试
    
    示例:
        >>> x = Variable(Tensor([[1.0, 2.0]]), name="x")
        >>> y = x * 2 + 1
        >>> y.backward()
        >>> print(x.grad)
    """
    
    def __init__(self, value: Tensor, name: str = None, requires_grad: bool = True):
        """初始化一个 Variable。
        
        参数:
            value: 张量值
            name: 变量名称（可选）
            requires_grad: 是否跟踪梯度
        """
        if not isinstance(value, Tensor):
            raise TypeError(f"Variable value must be Tensor, got {type(value)}")
        
        self.value = value
        self.grad = None
        self.creator = None
        self.requires_grad = requires_grad
        self.name = name if name else f"var_{id(self)}"
    
    def backward(self, retain_graph: bool = False):
        """通过反向传播计算梯度。
        
        这采用拓扑排序对计算图实现反向模式自动微分。
        
        参数:
            retain_graph: 是否保留计算图
        """
        if not self.requires_grad:
            return
        
        # 初始化梯度为 1（用于标量输出）
        if self.grad is None:
            self.grad = Tensor.ones(self.value.shape, self.value.dtype)
        
        # 拓扑排序获取执行顺序
        topo_order = []
        visited = set()
        
        def build_topo(var):
            if var not in visited and var.creator is not None:
                visited.add(var)
                for input_var in var.creator.inputs:
                    build_topo(input_var)
                topo_order.append(var)
        
        build_topo(self)
        
        # 按逆拓扑顺序进行反向传播
        for var in reversed(topo_order):
            if var.creator is None:
                continue
            
            # 获取该变量输出的梯度
            grad_output = var.grad
            
            # 计算输入的梯度
            grad_inputs = var.creator.backward(grad_output)
            
            # 将梯度累加到输入变量
            for input_var, grad_input in zip(var.creator.inputs, grad_inputs):
                if input_var.requires_grad:
                    if input_var.grad is None:
                        input_var.grad = grad_input
                    else:
                        # 累加梯度（对于多次使用的变量）
                        input_var.grad = input_var.grad.add(grad_input)
        
        # 如果不保留计算图，则清理它
        if not retain_graph:
            self.unchain_backward()
    
    def unchain_backward(self):
        """释放计算图以释放内存。"""
        def clear_graph(var):
            if var.creator is not None:
                for input_var in var.creator.inputs:
                    clear_graph(input_var)
                var.creator = None
        
        clear_graph(self)
    
    def clear_grad(self):
        """清除梯度。"""
        self.grad = None
    
    def detach(self) -> 'Variable':
        """创建一个从计算图中分离的新 Variable。
        
        返回:
            具有相同值但不跟踪梯度的新 Variable
        """
        return Variable(self.value.copy(), self.name + "_detached", requires_grad=False)
    
    # ==================== 算术运算 ====================
    
    def add(self, other) -> 'Variable':
        """加法运算。"""
        from tinytorch.autograd.ops.basic import Add
        
        if isinstance(other, (int, float)):
            other = Variable(Tensor([other]), requires_grad=False)
        
        return Add()(self, other)
    
    def sub(self, other) -> 'Variable':
        """减法运算。"""
        from tinytorch.autograd.ops.basic import Sub
        
        if isinstance(other, (int, float)):
            other = Variable(Tensor([other]), requires_grad=False)
        
        return Sub()(self, other)
    
    def mul(self, other) -> 'Variable':
        """乘法运算。"""
        from tinytorch.autograd.ops.basic import Mul
        
        if isinstance(other, (int, float)):
            other = Variable(Tensor([other]), requires_grad=False)
        
        return Mul()(self, other)
    
    def div(self, other) -> 'Variable':
        """除法运算。"""
        from tinytorch.autograd.ops.basic import Div
        
        if isinstance(other, (int, float)):
            other = Variable(Tensor([other]), requires_grad=False)
        
        return Div()(self, other)
    
    def neg(self) -> 'Variable':
        """取负运算。"""
        from tinytorch.autograd.ops.basic import Neg
        return Neg()(self)
    
    def pow(self, exponent: float) -> 'Variable':
        """幂运算。"""
        from tinytorch.autograd.ops.math_ops import Pow
        return Pow(exponent)(self)
    
    # ==================== 数学运算 ====================
    
    def exp(self) -> 'Variable':
        """指数运算。"""
        from tinytorch.autograd.ops.math_ops import Exp
        return Exp()(self)
    
    def log(self) -> 'Variable':
        """自然对数运算。"""
        from tinytorch.autograd.ops.math_ops import Log
        return Log()(self)
    
    def sqrt(self) -> 'Variable':
        """平方根运算。"""
        from tinytorch.autograd.ops.math_ops import Sqrt
        return Sqrt()(self)
    
    # ==================== 矩阵运算 ====================
    
    def matmul(self, other: 'Variable') -> 'Variable':
        """矩阵乘法。"""
        from tinytorch.autograd.ops.matrix import MatMul
        return MatMul()(self, other)
    
    def transpose(self, axes=None) -> 'Variable':
        """转置运算。"""
        from tinytorch.autograd.ops.matrix import Transpose
        return Transpose(axes)(self)
    
    def reshape(self, new_shape) -> 'Variable':
        """重塑形状运算。"""
        from tinytorch.autograd.ops.matrix import Reshape
        return Reshape(new_shape)(self)
    
    def sum(self, axis=None, keepdims=False) -> 'Variable':
        """求和约简。"""
        from tinytorch.autograd.ops.reduce import Sum
        return Sum(axis, keepdims)(self)
    
    def mean(self, axis=None, keepdims=False) -> 'Variable':
        """求均值约简。"""
        from tinytorch.autograd.ops.reduce import Mean
        return Mean(axis, keepdims)(self)
    
    # ==================== 激活函数 ====================
    
    def relu(self) -> 'Variable':
        """ReLU 激活函数。"""
        from tinytorch.autograd.ops.activation import ReLU
        return ReLU()(self)
    
    def sigmoid(self) -> 'Variable':
        """Sigmoid 激活函数。"""
        from tinytorch.autograd.ops.activation import Sigmoid
        return Sigmoid()(self)
    
    def tanh(self) -> 'Variable':
        """Tanh 激活函数。"""
        from tinytorch.autograd.ops.activation import Tanh
        return Tanh()(self)
    
    # ==================== 属性 ====================
    
    @property
    def shape(self) -> Shape:
        """获取张量形状。"""
        return self.value.shape
    
    @property
    def ndim(self) -> int:
        """获取维数。"""
        return self.value.shape.ndim
    
    @property
    def size(self) -> int:
        """获取元素总数。"""
        return self.value.shape.size
    
    @property
    def dtype(self) -> str:
        """获取数据类型。"""
        return self.value.dtype
    
    # ==================== 运算符重载 ====================
    
    def __add__(self, other):
        """加法运算符。"""
        return self.add(other)
    
    def __radd__(self, other):
        """右加法运算符。"""
        return self.add(other)
    
    def __sub__(self, other):
        """减法运算符。"""
        return self.sub(other)
    
    def __rsub__(self, other):
        """右减法运算符。"""
        if isinstance(other, (int, float)):
            other = Variable(Tensor([other]), requires_grad=False)
        return other.sub(self)
    
    def __mul__(self, other):
        """乘法运算符。"""
        return self.mul(other)
    
    def __rmul__(self, other):
        """右乘法运算符。"""
        return self.mul(other)
    
    def __truediv__(self, other):
        """除法运算符。"""
        return self.div(other)
    
    def __rtruediv__(self, other):
        """右除法运算符。"""
        if isinstance(other, (int, float)):
            other = Variable(Tensor([other]), requires_grad=False)
        return other.div(self)
    
    def __neg__(self):
        """取负运算符。"""
        return self.neg()
    
    def __pow__(self, exponent):
        """幂运算符。"""
        return self.pow(exponent)
    
    def __matmul__(self, other):
        """矩阵乘法运算符。"""
        return self.matmul(other)
    
    def __repr__(self) -> str:
        """字符串表示。"""
        grad_str = f", grad={self.grad}" if self.grad is not None else ""
        return f"Variable(name={self.name}, shape={self.value.shape}, requires_grad={self.requires_grad}{grad_str})"
    
    def __str__(self) -> str:
        """字符串表示。"""
        return self.__repr__()
