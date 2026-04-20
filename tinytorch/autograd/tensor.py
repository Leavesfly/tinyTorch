"""自动微分的变量类。

本模块提供了 Tensor 类，它包装了 NdArray 并通过动态计算图实现自动梯度计算。

Author: TinyAI Team
Version: 0.1.0
"""

from typing import Optional, Union
from tinytorch.ndarr import NdArray, Shape


class Tensor:
    _grad_enabled = True

    """自动微分变量。
    
    Tensor 包装了一个 NdArray，并为自动微分维护梯度信息和计算图连接。
    
    属性:
        value: 张量值
        grad: 梯度张量（与 value 形状相同）
        creator: 创建此变量的函数
        requires_grad: 是否计算梯度
        name: 变量名称，用于调试
    
    示例:
        >>> x = Tensor(NdArray([[1.0, 2.0]]), name="x")
        >>> y = x * 2 + 1
        >>> y.backward()
        >>> print(x.grad)
    """
    
    def __init__(self, value: NdArray, name: str = None, requires_grad: bool = True):
        """初始化一个 Tensor。
        
        参数:
            value: 张量值
            name: 变量名称（可选）
            requires_grad: 是否跟踪梯度
        """
        if not isinstance(value, NdArray):
            raise TypeError(f"Tensor value must be NdArray, got {type(value)}")
        
        self.value = value
        self.grad = None
        self.creator = None
        self.requires_grad = requires_grad
        self.name = name if name else f"var_{id(self)}"
    
    def backward(self, grad_output: Optional[Union['Tensor', NdArray]] = None, retain_graph: bool = False):
        """通过反向传播计算梯度。
        
        这采用拓扑排序对计算图实现反向模式自动微分。
        
        参数:
            grad_output: 输出梯度。标量输出可省略；非标量输出必须显式提供。
            retain_graph: 是否保留计算图
        """
        if not self.requires_grad:
            return
        
        if grad_output is not None:
            if isinstance(grad_output, Tensor):
                grad_output = grad_output.value
            if not isinstance(grad_output, NdArray):
                raise TypeError(f"grad_output must be NdArray or Tensor, got {type(grad_output)}")
            if grad_output.shape != self.value.shape:
                raise ValueError(
                    f"grad_output shape {grad_output.shape.dims} does not match output shape "
                    f"{self.value.shape.dims}"
                )
            if self.grad is None:
                self.grad = grad_output
            else:
                self.grad = self.grad.add(grad_output)
        else:
            # 仅标量输出可默认使用全 1 梯度
            if self.value.shape.size != 1:
                raise ValueError(
                    "grad_output must be provided for non-scalar outputs "
                    f"(got shape {self.value.shape.dims})"
                )
            if self.grad is None:
                self.grad = NdArray.ones(self.value.shape, self.value.dtype)
        
        # 迭代式拓扑排序，避免深图递归栈溢出
        topo_order = []
        # visit_state 字典记录节点的访问状态：
        # 0: 未访问 - 节点尚未被处理
        # 1: 访问中 - 节点正在被处理（已在栈中，正在遍历其输入）
        # 2: 已完成 - 节点及其所有输入都已处理完毕
        visit_state = {}  # 0: 未访问, 1: 访问中, 2: 已完成
        stack = [self]

        while stack:
            var = stack[-1]
            state = visit_state.get(var, 0)

            if state == 0:
                visit_state[var] = 1
                if var.creator is not None:
                    for input_var in var.creator.inputs:
                        if visit_state.get(input_var, 0) == 0:
                            stack.append(input_var)
            elif state == 1:
                stack.pop()
                visit_state[var] = 2
                if var.creator is not None:
                    topo_order.append(var)
            else:
                stack.pop()
        
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
        """释放计算图以释放内存。

        使用 visited 集合跟踪已访问的 Tensor 和 Function，
        确保多个输出指向同一 creator 时不会遗漏或重复清理。
        """
        stack = [self]
        visited_tensors = set()
        visited_functions = set()
        while stack:
            var = stack.pop()
            if id(var) in visited_tensors:
                continue
            visited_tensors.add(id(var))
            if var.creator is not None:
                func_id = id(var.creator)
                if func_id not in visited_functions:
                    visited_functions.add(func_id)
                    var.creator.clear_saved_tensors()
                    stack.extend(var.creator.inputs)
                var.creator = None
    
    def clear_grad(self):
        """清除梯度。"""
        self.grad = None
    
    def detach(self) -> 'Tensor':
        """创建一个从计算图中分离的新 Tensor。
        
        返回:
            具有相同值但不跟踪梯度的新 Tensor
        """
        return Tensor(self.value.copy(), self.name + "_detached", requires_grad=False)
    
    # ==================== 辅助方法 ====================

    @staticmethod
    def _ensure_tensor(value: Union["Tensor", int, float]) -> 'Tensor':
        """将标量值转换为不跟踪梯度的 Tensor。

        如果 value 已经是 Tensor 则直接返回，否则将 int/float 包装为 Tensor。
        """
        if isinstance(value, Tensor):
            return value
        if isinstance(value, NdArray):
            return Tensor(value, requires_grad=False)
        if isinstance(value, (int, float)):
            return Tensor(NdArray([value]), requires_grad=False)
        raise TypeError(f"Cannot convert {type(value)} to Tensor")

    # ==================== 算术运算 ====================
    
    def add(self, other: Union["Tensor", int, float]) -> 'Tensor':
        """加法运算。"""
        from tinytorch.autograd.ops.basic import Add
        other = self._ensure_tensor(other)
        return Add()(self, other)
    
    def sub(self, other: Union["Tensor", int, float]) -> 'Tensor':
        """减法运算。"""
        from tinytorch.autograd.ops.basic import Sub
        other = self._ensure_tensor(other)
        return Sub()(self, other)
    
    def mul(self, other: Union["Tensor", int, float]) -> 'Tensor':
        """乘法运算。"""
        from tinytorch.autograd.ops.basic import Mul
        other = self._ensure_tensor(other)
        return Mul()(self, other)
    
    def div(self, other: Union["Tensor", int, float]) -> 'Tensor':
        """除法运算。"""
        from tinytorch.autograd.ops.basic import Div
        other = self._ensure_tensor(other)
        return Div()(self, other)
    
    def neg(self) -> 'Tensor':
        """取负运算。"""
        from tinytorch.autograd.ops.basic import Neg
        return Neg()(self)
    
    def pow(self, exponent: float) -> 'Tensor':
        """幂运算。"""
        from tinytorch.autograd.ops.math_ops import Pow
        return Pow(exponent)(self)
    
    # ==================== 数学运算 ====================
    
    def exp(self) -> 'Tensor':
        """指数运算。"""
        from tinytorch.autograd.ops.math_ops import Exp
        return Exp()(self)
    
    def log(self) -> 'Tensor':
        """自然对数运算。"""
        from tinytorch.autograd.ops.math_ops import Log
        return Log()(self)
    
    def sqrt(self) -> 'Tensor':
        """平方根运算。"""
        from tinytorch.autograd.ops.math_ops import Sqrt
        return Sqrt()(self)
    
    # ==================== 矩阵运算 ====================
    
    def matmul(self, other: 'Tensor') -> 'Tensor':
        """矩阵乘法。"""
        from tinytorch.autograd.ops.matrix import MatMul
        return MatMul()(self, other)
    
    def transpose(self, axes=None) -> 'Tensor':
        """转置运算。"""
        from tinytorch.autograd.ops.matrix import Transpose
        return Transpose(axes)(self)
    
    def reshape(self, new_shape) -> 'Tensor':
        """重塑形状运算。"""
        from tinytorch.autograd.ops.matrix import Reshape
        return Reshape(new_shape)(self)
    
    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        """求和约简。"""
        from tinytorch.autograd.ops.reduce import Sum
        return Sum(axis, keepdims)(self)
    
    def mean(self, axis=None, keepdims=False) -> 'Tensor':
        """求均值约简。"""
        from tinytorch.autograd.ops.reduce import Mean
        return Mean(axis, keepdims)(self)
    
    # ==================== 激活函数 ====================
    
    def relu(self) -> 'Tensor':
        """ReLU 激活函数。"""
        from tinytorch.autograd.ops.activation import ReLU
        return ReLU()(self)
    
    def sigmoid(self) -> 'Tensor':
        """Sigmoid 激活函数。"""
        from tinytorch.autograd.ops.activation import Sigmoid
        return Sigmoid()(self)
    
    def tanh(self) -> 'Tensor':
        """Tanh 激活函数。"""
        from tinytorch.autograd.ops.activation import Tanh
        return Tanh()(self)
    
    def leaky_relu(self, negative_slope: float = 0.01) -> 'Tensor':
        """LeakyReLU 激活函数。"""
        from tinytorch.autograd.ops.activation import LeakyReLU
        return LeakyReLU(negative_slope=negative_slope)(self)
    
    # ==================== 属性 ====================
    
    @property
    def shape(self) -> Shape:
        """获取张量形状。"""
        return self.value.shape
    
    @property
    def data(self) -> list:
        """获取张量的数据列表（局平）。"""
        return self.value.data
    
    @data.setter
    def data(self, value: list):
        """设置张量的数据列表。"""
        self.value.data = value
    
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
        other = self._ensure_tensor(other)
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
        other = self._ensure_tensor(other)
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
        return f"Tensor(name={self.name}, shape={self.value.shape}, requires_grad={self.requires_grad}{grad_str})"
    
    def __str__(self) -> str:
        """字符串表示。"""
        return self.__repr__()


class _NoGrad:
    """禁用自动梯度追踪的上下文管理器。
    
    此上下文管理器用于临时禁用梯度计算，适用于推理阶段或不需要梯度的计算。
    
    使用方式:
        >>> with no_grad():
        ...     # 在此块内的所有操作都不会跟踪梯度
        ...     y = x * 2
    
    注意:
        - 此上下文管理器可以嵌套使用
        - 退出上下文后会恢复之前的梯度追踪状态
        - 不影响 requires_grad=False 的 Tensor
    """

    def __enter__(self):
        self._previous = Tensor._grad_enabled
        Tensor._grad_enabled = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        Tensor._grad_enabled = self._previous
        return False


def no_grad():
    """返回禁用梯度追踪的上下文管理器。"""
    return _NoGrad()