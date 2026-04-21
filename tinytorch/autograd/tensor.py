"""自动微分的变量类。

本模块提供了 Tensor 类，它包装了 NdArray 并通过动态计算图实现自动梯度计算。

Author: TinyAI Team
Version: 0.1.0
"""

from typing import Optional, Union
from tinytorch.ndarr import NdArray, Shape

# 算子模块路径注册表，集中管理所有算子的模块位置
_OPS_BASIC = 'tinytorch.autograd.ops.basic'
_OPS_MATH = 'tinytorch.autograd.ops.math_ops'
_OPS_MATRIX = 'tinytorch.autograd.ops.matrix'
_OPS_REDUCE = 'tinytorch.autograd.ops.reduce'
_OPS_ACTIVATION = 'tinytorch.autograd.ops.activation'

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
    
    def __init__(self, value: NdArray, name: Optional[str] = None, requires_grad: bool = True):
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

        采用拓扑排序对计算图实现反向模式自动微分。

        参数:
            grad_output: 输出梯度。标量输出可省略；非标量输出必须显式提供。
            retain_graph: 是否保留计算图
        """
        if not self.requires_grad:
            return

        self._init_grad(grad_output)
        topo_order = self._topological_sort()
        self._propagate_gradients(topo_order)

        if not retain_graph:
            self.unchain_backward()

    def _init_grad(self, grad_output: Optional[Union['Tensor', NdArray]]) -> None:
        """初始化输出节点的梯度。

        参数:
            grad_output: 外部提供的梯度；标量输出可为 None（默认全 1）。
        """
        if grad_output is not None:
            if isinstance(grad_output, Tensor):
                grad_output = grad_output.value
            if not isinstance(grad_output, NdArray):
                raise TypeError(f"grad_output must be NdArray or Tensor, got {type(grad_output)}")
            if grad_output.shape != self.value.shape:
                raise ValueError(
                    f"grad_output shape {grad_output.shape.dims} does not match "
                    f"output shape {self.value.shape.dims}"
                )
            self.grad = grad_output if self.grad is None else self.grad.add(grad_output)
        else:
            if self.value.shape.size != 1:
                raise ValueError(
                    "grad_output must be provided for non-scalar outputs "
                    f"(got shape {self.value.shape.dims})"
                )
            if self.grad is None:
                self.grad = NdArray.ones(self.value.shape, self.value.dtype)

    def _topological_sort(self) -> list['Tensor']:
        """对计算图进行迭代式拓扑排序，避免深图递归栈溢出。

        Returns:
            按拓扑顺序排列的、拥有 creator 的 Tensor 列表
        """
        _UNVISITED, _IN_PROGRESS, _DONE = 0, 1, 2

        topo_order = []
        visit_state = {}
        stack = [self]

        while stack:
            node = stack[-1]
            state = visit_state.get(node, _UNVISITED)

            if state == _UNVISITED:
                visit_state[node] = _IN_PROGRESS
                if node.creator is not None:
                    for input_node in node.creator.inputs:
                        if visit_state.get(input_node, _UNVISITED) == _UNVISITED:
                            stack.append(input_node)
            elif state == _IN_PROGRESS:
                stack.pop()
                visit_state[node] = _DONE
                if node.creator is not None:
                    topo_order.append(node)
            else:
                stack.pop()

        return topo_order

    @staticmethod
    def _propagate_gradients(topo_order: list['Tensor']) -> None:
        """按逆拓扑顺序将梯度从输出传播到输入。

        参数:
            topo_order: _topological_sort 返回的有序节点列表
        """
        for node in reversed(topo_order):
            if node.creator is None:
                continue

            grad_inputs = node.creator.backward(node.grad)

            for input_node, grad_input in zip(node.creator.inputs, grad_inputs):
                if input_node.requires_grad:
                    if input_node.grad is None:
                        input_node.grad = grad_input
                    else:
                        input_node.grad = input_node.grad.add(grad_input)
    
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

    def _apply_binary_op(self, module_path: str, op_name: str, other: Union["Tensor", int, float]) -> 'Tensor':
        """执行二元运算的通用辅助方法。

        Args:
            module_path: 运算类所在的模块路径，如 ``'tinytorch.autograd.ops.basic'``
            op_name: 运算类名，如 ``'Add'``
            other: 另一个操作数

        Returns:
            运算结果 Tensor
        """
        import importlib
        op_class = getattr(importlib.import_module(module_path), op_name)
        other = self._ensure_tensor(other)
        return op_class()(self, other)

    def _apply_unary_op(self, module_path: str, op_name: str, **kwargs) -> 'Tensor':
        """执行一元运算的通用辅助方法。

        Args:
            module_path: 运算类所在的模块路径
            op_name: 运算类名
            **kwargs: 传递给运算类构造函数的参数

        Returns:
            运算结果 Tensor
        """
        import importlib
        op_class = getattr(importlib.import_module(module_path), op_name)
        return op_class(**kwargs)(self)

    def add(self, other: Union["Tensor", int, float]) -> 'Tensor':
        """加法运算。"""
        return self._apply_binary_op(_OPS_BASIC, 'Add', other)

    def sub(self, other: Union["Tensor", int, float]) -> 'Tensor':
        """减法运算。"""
        return self._apply_binary_op(_OPS_BASIC, 'Sub', other)

    def mul(self, other: Union["Tensor", int, float]) -> 'Tensor':
        """乘法运算。"""
        return self._apply_binary_op(_OPS_BASIC, 'Mul', other)

    def div(self, other: Union["Tensor", int, float]) -> 'Tensor':
        """除法运算。"""
        return self._apply_binary_op(_OPS_BASIC, 'Div', other)

    def neg(self) -> 'Tensor':
        """取负运算。"""
        return self._apply_unary_op(_OPS_BASIC, 'Neg')

    def pow(self, exponent: float) -> 'Tensor':
        """幂运算。"""
        from tinytorch.autograd.ops.math_ops import Pow
        return Pow(exponent)(self)

    def exp(self) -> 'Tensor':
        """指数运算。"""
        return self._apply_unary_op(_OPS_MATH, 'Exp')

    def log(self) -> 'Tensor':
        """自然对数运算。"""
        return self._apply_unary_op(_OPS_MATH, 'Log')

    def sqrt(self) -> 'Tensor':
        """平方根运算。"""
        return self._apply_unary_op(_OPS_MATH, 'Sqrt')

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """矩阵乘法。"""
        return self._apply_binary_op(_OPS_MATRIX, 'MatMul', other)

    def transpose(self, axes: Optional[tuple] = None) -> 'Tensor':
        """转置运算。"""
        from tinytorch.autograd.ops.matrix import Transpose
        return Transpose(axes)(self)

    def reshape(self, new_shape: tuple) -> 'Tensor':
        """重塑形状运算。"""
        from tinytorch.autograd.ops.matrix import Reshape
        return Reshape(new_shape)(self)

    def sum(self, axis: Optional[Union[int, tuple]] = None, keepdims: bool = False) -> 'Tensor':
        """求和约简。"""
        return self._apply_unary_op(_OPS_REDUCE, 'Sum', axis=axis, keepdims=keepdims)

    def mean(self, axis: Optional[Union[int, tuple]] = None, keepdims: bool = False) -> 'Tensor':
        """求均值约简。"""
        return self._apply_unary_op(_OPS_REDUCE, 'Mean', axis=axis, keepdims=keepdims)

    def relu(self) -> 'Tensor':
        """ReLU 激活函数。"""
        return self._apply_unary_op(_OPS_ACTIVATION, 'ReLU')

    def sigmoid(self) -> 'Tensor':
        """Sigmoid 激活函数。"""
        return self._apply_unary_op(_OPS_ACTIVATION, 'Sigmoid')

    def tanh(self) -> 'Tensor':
        """Tanh 激活函数。"""
        return self._apply_unary_op(_OPS_ACTIVATION, 'Tanh')

    def leaky_relu(self, negative_slope: float = 0.01) -> 'Tensor':
        """LeakyReLU 激活函数。"""
        return self._apply_unary_op(_OPS_ACTIVATION, 'LeakyReLU', negative_slope=negative_slope)

    @property
    def shape(self) -> Shape:
        """获取张量形状。"""
        return self.value.shape
    
    @property
    def data(self) -> list:
        """获取张量的扁平数据列表。"""
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
    
    def __add__(self, other: Union["Tensor", int, float]) -> 'Tensor':
        """加法运算符。"""
        return self.add(other)
    
    def __radd__(self, other: Union["Tensor", int, float]) -> 'Tensor':
        """右加法运算符。"""
        return self.add(other)
    
    def __sub__(self, other: Union["Tensor", int, float]) -> 'Tensor':
        """减法运算符。"""
        return self.sub(other)
    
    def __rsub__(self, other: Union["Tensor", int, float]) -> 'Tensor':
        """右减法运算符。"""
        other = self._ensure_tensor(other)
        return other.sub(self)
    
    def __mul__(self, other: Union["Tensor", int, float]) -> 'Tensor':
        """乘法运算符。"""
        return self.mul(other)
    
    def __rmul__(self, other: Union["Tensor", int, float]) -> 'Tensor':
        """右乘法运算符。"""
        return self.mul(other)
    
    def __truediv__(self, other: Union["Tensor", int, float]) -> 'Tensor':
        """除法运算符。"""
        return self.div(other)
    
    def __rtruediv__(self, other: Union["Tensor", int, float]) -> 'Tensor':
        """右除法运算符。"""
        other = self._ensure_tensor(other)
        return other.div(self)
    
    def __neg__(self) -> 'Tensor':
        """取负运算符。"""
        return self.neg()
    
    def __pow__(self, exponent: Union[int, float]) -> 'Tensor':
        """幂运算符。"""
        return self.pow(exponent)
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """矩阵乘法运算符。"""
        return self.matmul(other)
    
    def __repr__(self) -> str:
        """字符串表示。"""
        grad_str = f", grad={self.grad}" if self.grad is not None else ""
        return f"Tensor(name={self.name}, shape={self.value.shape}, requires_grad={self.requires_grad}{grad_str})"

    __str__ = __repr__


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