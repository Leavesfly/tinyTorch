"""基本算术运算 - 支持自动微分。

本模块实现以下基本算术运算:
    - Add:  元素级加法 z = x + y
    - Sub:  元素级减法 z = x - y
    - Mul:  元素级乘法 z = x * y
    - Div:  元素级除法 z = x / y
    - Neg:  取负运算 y = -x

所有运算均支持广播机制，反向传播时自动处理梯度归约。
"""

from typing import List
from tinytorch.ndarr import NdArray
from tinytorch.autograd.function import Function
from tinytorch.autograd.ops.utils import sum_to_shape


class Add(Function):
    """元素级加法运算。

    数学表达式: z = x + y

    前向传播:
        z = x + y

    反向传播:
        ∂L/∂x = ∂L/∂z
        ∂L/∂y = ∂L/∂z

    注意: 支持广播机制，反向传播时自动归约梯度至输入形状。
    """

    def forward(self, x: NdArray, y: NdArray) -> NdArray:
        """前向传播: 计算元素级加法。"""
        return x.add(y)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。

        梯度均等地流向两个输入，并处理广播维度的归约。
        """
        x_shape, y_shape = self.inputs[0].value.shape, self.inputs[1].value.shape
        grad_x = sum_to_shape(grad_output, x_shape) if grad_output.shape != x_shape else grad_output
        grad_y = sum_to_shape(grad_output, y_shape) if grad_output.shape != y_shape else grad_output
        return [grad_x, grad_y]


class Sub(Function):
    """元素级减法运算。

    数学表达式: z = x - y

    前向传播:
        z = x - y

    反向传播:
        ∂L/∂x = ∂L/∂z
        ∂L/∂y = -∂L/∂z

    注意: 支持广播机制，反向传播时自动归约梯度至输入形状。
    """

    def forward(self, x: NdArray, y: NdArray) -> NdArray:
        """前向传播: 计算元素级减法。"""
        return x.sub(y)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。"""
        x_shape, y_shape = self.inputs[0].value.shape, self.inputs[1].value.shape
        grad_x = sum_to_shape(grad_output, x_shape) if grad_output.shape != x_shape else grad_output
        grad_y = sum_to_shape(grad_output.neg(), y_shape) if grad_output.shape != y_shape else grad_output.neg()
        return [grad_x, grad_y]


class Mul(Function):
    """元素级乘法运算。

    数学表达式: z = x * y

    前向传播:
        z = x * y

    反向传播:
        ∂L/∂x = ∂L/∂z * y
        ∂L/∂y = ∂L/∂z * x

    注意: 支持广播机制，反向传播时自动归约梯度至输入形状。
    """

    def forward(self, x: NdArray, y: NdArray) -> NdArray:
        """前向传播: 计算元素级乘法。"""
        self.save_for_backward(x, y)
        return x.mul(y)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。"""
        x, y = self.get_saved_tensors()
        x_shape, y_shape = self.inputs[0].value.shape, self.inputs[1].value.shape
        grad_x = sum_to_shape(grad_output.mul(y), x_shape) if grad_output.shape != x_shape else grad_output.mul(y)
        grad_y = sum_to_shape(grad_output.mul(x), y_shape) if grad_output.shape != y_shape else grad_output.mul(x)
        return [grad_x, grad_y]


class Div(Function):
    """元素级除法运算。

    数学表达式: z = x / y

    前向传播:
        z = x / y

    反向传播:
        ∂L/∂x = ∂L/∂z / y
        ∂L/∂y = -∂L/∂z * x / y²

    注意:
        - 支持广播机制，反向传播时自动归约梯度至输入形状
        - 添加小常数 eps 保证数值稳定性，避免除零错误
    """

    def forward(self, x: NdArray, y: NdArray) -> NdArray:
        """前向传播: 计算元素级除法。"""
        self.save_for_backward(x, y)
        return x.div(y)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。"""
        x, y = self.get_saved_tensors()
        x_shape, y_shape = self.inputs[0].value.shape, self.inputs[1].value.shape

        # 数值稳定性: 避免 y² 接近 0 时梯度爆炸
        eps = 1e-10
        y_sq_safe = y.pow(2).add(eps)
        grad_x = grad_output.div(y)
        grad_y = grad_output.neg().mul(x).div(y_sq_safe)

        # 处理广播
        grad_x = sum_to_shape(grad_x, x_shape) if grad_x.shape != x_shape else grad_x
        grad_y = sum_to_shape(grad_y, y_shape) if grad_y.shape != y_shape else grad_y
        return [grad_x, grad_y]


class Neg(Function):
    """取负运算。

    数学表达式: y = -x

    前向传播:
        y = -x

    反向传播:
        ∂L/∂x = -∂L/∂y
    """

    def forward(self, x: NdArray) -> NdArray:
        """前向传播: 计算取负。"""
        return x.neg()

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入梯度。"""
        return [grad_output.neg()]
