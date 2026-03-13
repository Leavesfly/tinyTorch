"""归约运算 - 支持自动微分。

本模块实现以下归约运算:
    - Sum:  求和归约 y = sum(x, axis, keepdims)
    - Mean: 平均归约 y = mean(x, axis, keepdims)

归约运算在反向传播时需要将梯度广播回原始形状。
"""

from typing import List, Optional
from tinytorch.ndarr import NdArray
from tinytorch.autograd.function import Function


class Sum(Function):
    """求和归约运算。

    数学表达式: y = sum(x, axis, keepdims)

    前向传播:
        y = sum(x, axis, keepdims)

    反向传播:
        ∂L/∂x = broadcast(∂L/∂y) to x.shape

    参数:
        axis: 归约轴，None 表示对所有维度归约
        keepdims: 是否保留归约后的维度 (大小为 1)

    示例:
        >>> x = [[1, 2], [3, 4]]
        >>> Sum(axis=0)(x)  # [4, 6]
        >>> Sum(axis=1)(x)  # [3, 7]
        >>> Sum()(x)        # 10
    """

    def __init__(self, axis: Optional[int] = None, keepdims: bool = False):
        """初始化求和归约。

        Args:
            axis: 归约轴，None 表示对所有维度归约
            keepdims: 是否保留归约后的维度
        """
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: NdArray) -> NdArray:
        """前向传播: 计算求和归约。"""
        self.input_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。

        将梯度广播回输入形状。根据 keepdims 参数采取不同策略:
        - keepdims=True: 直接广播
        - keepdims=False: 先扩展归约维度再广播
        """
        if self.keepdims:
            # 维度已保留，直接广播
            return [grad_output._broadcast_to(self.input_shape)]

        if self.axis is None:
            # 对所有维度归约: 梯度复制到所有位置
            grad_x = NdArray(
                [grad_output.data[0]] * self.input_shape.size,
                self.input_shape, grad_output.dtype
            )
            return [grad_x]

        # 对特定轴归约: 扩展维度后广播
        new_shape_list = list(grad_output.shape.dims)
        axis = self.axis if self.axis >= 0 else self.input_shape.ndim + self.axis
        new_shape_list.insert(axis, 1)
        grad_expanded = grad_output.reshape(tuple(new_shape_list))
        return [grad_expanded._broadcast_to(self.input_shape)]


class Mean(Function):
    """平均归约运算。

    数学表达式: y = mean(x, axis, keepdims)

    前向传播:
        y = mean(x, axis, keepdims)

    反向传播:
        ∂L/∂x = broadcast(∂L/∂y / count) to x.shape

    参数:
        axis: 归约轴，None 表示对所有维度归约
        keepdims: 是否保留归约后的维度 (大小为 1)

    示例:
        >>> x = [[1, 2], [3, 4]]
        >>> Mean(axis=0)(x)  # [2, 3]
        >>> Mean(axis=1)(x)  # [1.5, 3.5]
        >>> Mean()(x)        # 2.5
    """

    def __init__(self, axis: Optional[int] = None, keepdims: bool = False):
        """初始化平均归约。

        Args:
            axis: 归约轴，None 表示对所有维度归约
            keepdims: 是否保留归约后的维度
        """
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: NdArray) -> NdArray:
        """前向传播: 计算平均归约。"""
        self.input_shape = x.shape

        # 计算归约元素数量，用于反向传播时梯度缩放
        if self.axis is None:
            self.count = x.shape.size
        else:
            axis = self.axis if self.axis >= 0 else x.shape.ndim + self.axis
            self.count = x.shape.dims[axis]

        return x.mean(axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。

        梯度除以归约元素数量后广播回输入形状。
        """
        # 梯度缩放: 除以归约元素数量
        grad_scaled = grad_output.div(self.count)

        # 广播回输入形状 (与 Sum 相同的逻辑)
        if self.keepdims:
            return [grad_scaled._broadcast_to(self.input_shape)]

        if self.axis is None:
            grad_x = NdArray(
                [grad_scaled.data[0]] * self.input_shape.size,
                self.input_shape, grad_output.dtype
            )
            return [grad_x]

        new_shape_list = list(grad_scaled.shape.dims)
        axis = self.axis if self.axis >= 0 else self.input_shape.ndim + self.axis
        new_shape_list.insert(axis, 1)
        grad_expanded = grad_scaled.reshape(tuple(new_shape_list))
        return [grad_expanded._broadcast_to(self.input_shape)]
