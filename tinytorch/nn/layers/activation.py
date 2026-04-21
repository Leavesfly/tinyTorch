"""激活函数层封装。

将 autograd.ops.activation 中的激活函数封装为神经网络层。
使用 _SimpleActivation 基类消除重复的样板代码。

Author: TinyAI Team
"""

from tinytorch.autograd.tensor import Tensor
from tinytorch.nn.module import Module


class _SimpleActivation(Module):
    """无参数激活函数的通用基类。

    子类只需指定 _op_name 即可自动获得 forward 和 __repr__ 实现。
    """

    _op_name: str = ""  # 对应 Tensor 上的方法名，如 "relu"

    def __init__(self, name: str = None):
        super().__init__(name=name or self.__class__.__name__)

    def forward(self, input: Tensor) -> Tensor:
        """前向传播: 调用 Tensor 上对应的激活函数方法。"""
        return getattr(input, self._op_name)()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ReLU(_SimpleActivation):
    """ReLU 激活层。

    ReLU(x) = max(0, x)

    Example:
        >>> relu = ReLU()
        >>> x = Tensor(NdArray([[-1, 2], [3, -4]]))
        >>> y = relu(x)
        >>> print(y.value.to_list())
        [[0, 2], [3, 0]]
    """

    _op_name = "relu"


class Sigmoid(_SimpleActivation):
    """Sigmoid 激活层。

    Sigmoid(x) = 1 / (1 + exp(-x))

    Example:
        >>> sigmoid = Sigmoid()
        >>> x = Tensor(NdArray([[0, 1], [2, -1]]))
        >>> y = sigmoid(x)
    """

    _op_name = "sigmoid"


class Tanh(_SimpleActivation):
    """Tanh 激活层。

    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Example:
        >>> tanh = Tanh()
        >>> x = Tensor(NdArray([[0, 1], [2, -1]]))
        >>> y = tanh(x)
    """

    _op_name = "tanh"


class LeakyReLU(Module):
    """LeakyReLU 激活层。

    LeakyReLU(x) = max(negative_slope * x, x)

    Example:
        >>> leaky_relu = LeakyReLU(negative_slope=0.01)
        >>> x = Tensor(NdArray([[-1, 2], [3, -4]]))
        >>> y = leaky_relu(x)
    """

    def __init__(self, negative_slope: float = 0.01, name: str = None):
        """初始化 LeakyReLU 层。

        Args:
            negative_slope: 负半轴的斜率，默认为 0.01
            name: 层的名称
        """
        super().__init__(name=name or 'LeakyReLU')
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        """前向传播。"""
        return input.leaky_relu(self.negative_slope)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(negative_slope={self.negative_slope})"
