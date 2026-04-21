"""Dropout 层实现。

Author: TinyAI Team
"""

from tinytorch.nn.module import Module
from tinytorch.autograd.tensor import Tensor
from tinytorch.ndarr.ndarray import NdArray


class Dropout(Module):
    """Dropout 层。

    在训练时随机将输入的一部分元素置零，用于防止过拟合。
    在评估时不做任何操作。

    Attributes:
        p: dropout 概率（置零的概率）
        training: 训练/评估模式标志

    Example:
        >>> dropout = Dropout(p=0.5)
        >>> x = Tensor(NdArray.randn((32, 10)))
        >>> y = dropout(x)
    """

    def __init__(self, p: float = 0.5, name: str = None):
        """初始化 Dropout 层。

        Args:
            p: dropout 概率，范围 [0, 1)
            name: 层的名称
        """
        super().__init__(name=name or 'Dropout')

        if not 0 <= p < 1:
            raise ValueError(f"dropout probability has to be in [0, 1), got {p}")

        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        """前向传播。

        Args:
            input: 输入变量

        Returns:
            Dropout 后的输出变量
        """
        if not self.training or self.p == 0:
            return input

        # 批量生成随机数构建 dropout mask
        scale = 1.0 / (1.0 - self.p)
        rand_arr = NdArray.uniform(0.0, 1.0, input.value.shape.dims, dtype=input.value.dtype)
        mask_data = [scale if r >= self.p else 0.0 for r in rand_arr.data]
        mask = NdArray(mask_data, input.value.shape, input.value.dtype)
        mask_var = Tensor(mask, requires_grad=False)

        return input * mask_var

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
