"""矩阵与张量变换运算 - 支持自动微分。

本模块实现以下运算:
    - MatMul:    矩阵乘法 C = A @ B
    - Transpose: 张量转置 B = A^T
    - Reshape:   张量重塑 y = reshape(x, new_shape)
"""

from typing import List, Tuple, Union
from tinytorch.ndarr import NdArray
from tinytorch.autograd.function import Function


class MatMul(Function):
    """矩阵乘法运算。

    数学表达式: C = A @ B

    前向传播:
        C = A @ B

    反向传播:
        ∂L/∂A = ∂L/∂C @ B^T
        ∂L/∂B = A^T @ ∂L/∂C

    形状要求:
        - A: (M, K)
        - B: (K, N)
        - C: (M, N)
    """

    def forward(self, a: NdArray, b: NdArray) -> NdArray:
        """前向传播: 计算矩阵乘法。"""
        self.save_for_backward(a, b)
        return a.matmul(b)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。

        利用矩阵微分的链式法则。
        """
        a, b = self.get_saved_tensors()
        grad_a = grad_output.matmul(b.transpose())  # ∂L/∂A = ∂L/∂C @ B^T
        grad_b = a.transpose().matmul(grad_output)  # ∂L/∂B = A^T @ ∂L/∂C
        return [grad_a, grad_b]


class Transpose(Function):
    """张量转置运算。

    数学表达式: B = A^T (或按指定轴排列)

    前向传播:
        B = transpose(A, axes)

    反向传播:
        ∂L/∂A = transpose(∂L/∂B, inverse_axes)

    参数:
        axes: 维度排列顺序，None 表示反转所有维度

    示例:
        >>> x.shape = (2, 3, 4)
        >>> Transpose(axes=(0, 2, 1))(x).shape  # (2, 4, 3)
    """

    def __init__(self, axes: Tuple[int, ...] = None):
        """初始化转置运算。

        Args:
            axes: 维度排列顺序，None 表示默认反转所有维度
        """
        super().__init__()
        self.axes = axes

    def forward(self, x: NdArray) -> NdArray:
        """前向传播: 计算张量转置。"""
        return x.transpose(self.axes)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。

        通过逆排列恢复原始维度顺序。
        """
        if self.axes is None:
            # 默认转置: 反转所有维度
            return [grad_output.transpose()]

        # 计算逆排列
        inv_axes = [0] * len(self.axes)
        for i, ax in enumerate(self.axes):
            inv_axes[ax] = i
        return [grad_output.transpose(tuple(inv_axes))]


class Reshape(Function):
    """张量重塑运算。

    数学表达式: y = reshape(x, new_shape)

    前向传播:
        y = reshape(x, new_shape)

    反向传播:
        ∂L/∂x = reshape(∂L/∂y, old_shape)

    注意:
        - 新形状的元素总数必须与原形状相同
        - 重塑不改变数据存储顺序
    """

    def __init__(self, new_shape: Union[Tuple[int, ...], List[int]]):
        """初始化重塑运算。

        Args:
            new_shape: 目标形状，元素总数必须与输入相同
        """
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x: NdArray) -> NdArray:
        """前向传播: 计算张量重塑。"""
        self.old_shape = x.shape
        return x.reshape(self.new_shape)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。

        将梯度重塑回原始形状。
        """
        return [grad_output.reshape(self.old_shape.dims)]
