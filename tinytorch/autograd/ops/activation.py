"""激活函数运算 - 支持自动微分。

本模块实现以下激活函数:
    - ReLU:      修正线性单元 y = max(0, x)
    - Sigmoid:   S型函数 y = 1 / (1 + exp(-x))
    - Tanh:      双曲正切函数 y = tanh(x)
    - LeakyReLU: 泄漏修正线性单元 y = max(αx, x)
"""

from typing import List
from tinytorch.ndarr import NdArray
from tinytorch.autograd.function import Function


class ReLU(Function):
    """修正线性单元 (Rectified Linear Unit)。

    数学表达式: y = max(0, x)

    前向传播:
        y = max(0, x)

    反向传播:
        ∂L/∂x = ∂L/∂y * (x > 0)

    特点:
        - 计算高效，收敛速度快
        - 缓解梯度消失问题
        - 存在神经元死亡风险 (x < 0 时梯度为 0)
    """

    def forward(self, x: NdArray) -> NdArray:
        """前向传播: 计算 ReLU 激活。"""
        self.save_for_backward(x)
        return x.relu()

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。

        梯度在 x > 0 时为 1，否则为 0。
        """
        x, = self.get_saved_tensors()
        mask_data = [1.0 if val > 0 else 0.0 for val in x.data]
        mask = NdArray(mask_data, x.shape, x.dtype)
        return [grad_output.mul(mask)]


class Sigmoid(Function):
    """S型激活函数。

    数学表达式: y = 1 / (1 + exp(-x))

    前向传播:
        y = sigmoid(x)

    反向传播:
        ∂L/∂x = ∂L/∂y * y * (1 - y)

    特点:
        - 输出范围 (0, 1)，适合概率输出
        - 中心对称于 (0, 0.5)
        - 存在梯度消失问题 (|x| 较大时)
    """

    def forward(self, x: NdArray) -> NdArray:
        """前向传播: 计算 Sigmoid 激活。"""
        y = x.sigmoid()
        self.save_for_backward(y)
        return y

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。

        利用链式法则: dy/dx = y * (1 - y)
        """
        y, = self.get_saved_tensors()
        one = NdArray.ones(y.shape, y.dtype)
        return [grad_output.mul(y).mul(one.sub(y))]


class Tanh(Function):
    """双曲正切激活函数。

    数学表达式: y = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    前向传播:
        y = tanh(x)

    反向传播:
        ∂L/∂x = ∂L/∂y * (1 - y²)

    特点:
        - 输出范围 (-1, 1)，零中心
        - 收敛速度优于 Sigmoid
        - 仍存在梯度消失问题
    """

    def forward(self, x: NdArray) -> NdArray:
        """前向传播: 计算 Tanh 激活。"""
        y = x.tanh()
        self.save_for_backward(y)
        return y

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。

        利用链式法则: dy/dx = 1 - y²
        """
        y, = self.get_saved_tensors()
        one = NdArray.ones(y.shape, y.dtype)
        return [grad_output.mul(one.sub(y.pow(2)))]


class LeakyReLU(Function):
    """泄漏修正线性单元 (Leaky Rectified Linear Unit)。

    数学表达式: y = x (当 x > 0) 否则 y = α * x

    其中 α (negative_slope) 为负区间的斜率，通常取较小值如 0.01。

    前向传播:
        y = max(αx, x)

    反向传播:
        ∂L/∂x = ∂L/∂y * 1     (当 x > 0)
        ∂L/∂x = ∂L/∂y * α     (当 x ≤ 0)

    特点:
        - 解决 ReLU 的神经元死亡问题
        - 负区间保留微小梯度，允许信息流动
    """

    def __init__(self, negative_slope: float = 0.01):
        """初始化 LeakyReLU。

        Args:
            negative_slope: 负区间的斜率，默认 0.01
        """
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: NdArray) -> NdArray:
        """前向传播: 计算 LeakyReLU 激活。"""
        self.save_for_backward(x)
        data = [v if v > 0 else self.negative_slope * v for v in x.data]
        return NdArray(data, x.shape, x.dtype)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。

        根据输入值的正负选择不同的梯度缩放因子。
        """
        x, = self.get_saved_tensors()
        grad_scale = [1.0 if v > 0 else self.negative_slope for v in x.data]
        grad_mask = NdArray(grad_scale, x.shape, x.dtype)
        return [grad_output.mul(grad_mask)]
