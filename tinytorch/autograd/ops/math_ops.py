"""数学函数运算 - 支持自动微分。

本模块实现以下数学函数:
    - Exp:  指数函数 y = exp(x)
    - Log:  自然对数 y = ln(x)
    - Sqrt: 平方根 y = √x
    - Pow:  幂函数 y = x^n
"""

from typing import List
from tinytorch.ndarr import NdArray
from tinytorch.autograd.function import Function


class Exp(Function):
    """指数函数。

    数学表达式: y = exp(x)

    前向传播:
        y = exp(x)

    反向传播:
        ∂L/∂x = ∂L/∂y * exp(x) = ∂L/∂y * y

    特点:
        - 输出恒为正
        - 常用于 Softmax 和概率计算
    """

    def forward(self, x: NdArray) -> NdArray:
        """前向传播: 计算指数函数。"""
        y = x.exp()
        self.save_for_backward(y)
        return y

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。

        利用恒等式: d(exp(x))/dx = exp(x) = y
        """
        y, = self.get_saved_tensors()
        return [grad_output.mul(y)]


class Log(Function):
    """自然对数函数。

    数学表达式: y = ln(x)

    前向传播:
        y = ln(x)

    反向传播:
        ∂L/∂x = ∂L/∂y / x

    注意:
        - 输入 x 必须为正数
        - 添加小常数 epsilon 保证数值稳定性，避免除零错误
    """

    def __init__(self, epsilon: float = 1e-10):
        """初始化对数函数。

        Args:
            epsilon: 数值稳定性常数，防止除零错误
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: NdArray) -> NdArray:
        """前向传播: 计算自然对数。"""
        self.save_for_backward(x)
        return x.log()

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。

        使用 x + epsilon 替代 x 以避免除零错误。
        """
        x, = self.get_saved_tensors()
        x_safe = x.add(self.epsilon)
        return [grad_output.div(x_safe)]


class Sqrt(Function):
    """平方根函数。

    数学表达式: y = √x

    前向传播:
        y = √x

    反向传播:
        ∂L/∂x = ∂L/∂y / (2 * √x) = ∂L/∂y / (2 * y)

    注意:
        - 输入 x 必须非负
        - 添加小常数 epsilon 保证数值稳定性，避免 x=0 时除零
    """

    def __init__(self, epsilon: float = 1e-10):
        """初始化平方根函数。

        Args:
            epsilon: 数值稳定性常数，防止除零错误
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: NdArray) -> NdArray:
        """前向传播: 计算平方根。"""
        y = x.sqrt()
        self.save_for_backward(y)
        return y

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。

        分母添加 epsilon 以避免 x=0 时除零错误。
        """
        y, = self.get_saved_tensors()
        denom = y.mul(2.0).add(self.epsilon)
        return [grad_output.div(denom)]


class Pow(Function):
    """幂函数。

    数学表达式: y = x^n

    前向传播:
        y = x^n

    反向传播:
        ∂L/∂x = ∂L/∂y * n * x^(n-1)

    注意:
        - 当指数 n < 1 时，x^(n-1) 在 x=0 处可能发散
        - 添加小常数 epsilon 保证数值稳定性
    """

    def __init__(self, exponent: float, epsilon: float = 1e-10):
        """初始化幂函数。

        Args:
            exponent: 幂指数 n
            epsilon: 数值稳定性常数，当 n < 1 时避免 0^(n-1) 发散
        """
        super().__init__()
        self.exponent = exponent
        self.epsilon = epsilon

    def forward(self, x: NdArray) -> NdArray:
        """前向传播: 计算幂函数。"""
        self.save_for_backward(x)
        return x.pow(self.exponent)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。

        当指数 n < 1 时，对 x 添加 epsilon 以避免 0^(n-1) 发散。
        """
        x, = self.get_saved_tensors()
        exp_minus_1 = self.exponent - 1
        x_safe = x.add(self.epsilon) if exp_minus_1 < 0 else x
        grad = grad_output.mul(x_safe.pow(exp_minus_1).mul(self.exponent))
        return [grad]
