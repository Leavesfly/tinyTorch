"""层归一化（LayerNorm）层实现。

Author: TinyAI Team
"""

from tinytorch.nn.module import Module
from tinytorch.nn.parameter import Parameter
from tinytorch.autograd.tensor import Tensor
from tinytorch.ndarr.ndarray import NdArray


class LayerNorm(Module):
    """层归一化（Layer Normalization）。
    
    对输入的最后几个维度进行归一化，常用于 Transformer 等模型。
    
    公式: y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    Attributes:
        normalized_shape: 需要归一化的形状（通常是最后几个维度）
        eps: 数值稳定性常数
        weight: 缩放参数 gamma
        bias: 偏移参数 beta
    
    Example:
        >>> layer_norm = LayerNorm((10,))
        >>> x = Tensor(NdArray.randn((32, 10)))
        >>> y = layer_norm(x)
    """
    
    def __init__(self, normalized_shape: tuple, eps: float = 1e-5, 
                 elementwise_affine: bool = True, name: str = None):
        """初始化 LayerNorm 层。
        
        Args:
            normalized_shape: 需要归一化的形状
            eps: 数值稳定性常数
            elementwise_affine: 是否使用可学习的缩放和偏移参数
            name: 层的名称
        """
        super().__init__(name=name or 'LayerNorm')
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            # gamma 参数（缩放）
            self.weight = Parameter(NdArray.ones(normalized_shape),
                                    name=f'{self.name}.weight')
            # beta 参数（偏移）
            self.bias = Parameter(NdArray.zeros(normalized_shape),
                                  name=f'{self.name}.bias')
        else:
            self.weight = None
            self.bias = None
    
    def forward(self, input: Tensor) -> Tensor:
        """前向传播。
        
        Args:
            input: 输入变量
        
        Returns:
            归一化后的输出变量
        """
        input_dims = input.value.shape.dims
        if len(input_dims) < len(self.normalized_shape):
            raise ValueError(
                f"input dims {input_dims} must have at least "
                f"{len(self.normalized_shape)} dims for normalized_shape={self.normalized_shape}"
            )

        if tuple(input_dims[-len(self.normalized_shape):]) != tuple(self.normalized_shape):
            raise ValueError(
                f"Expected trailing dims {self.normalized_shape}, "
                f"got {input_dims[-len(self.normalized_shape):]}"
            )

        # 在最后若干维上做归一化
        reduce_axes = list(range(len(input_dims) - len(self.normalized_shape), len(input_dims)))

        mean = input
        for axis in reduce_axes:
            mean = mean.mean(axis=axis, keepdims=True)

        # 方差: var = mean((x - mean)^2)，与 mean 使用相同归约轴
        diff = input - mean
        variance = diff * diff
        for axis in reduce_axes:
            variance = variance.mean(axis=axis, keepdims=True)
        
        # 归一化: (x - mean) / sqrt(var + eps)
        std = (variance + Tensor(NdArray([self.eps]), requires_grad=False)).sqrt()
        normalized = diff / std
        
        # 应用仿射变换
        if self.elementwise_affine:
            output = normalized * self.weight + self.bias
        else:
            output = normalized
        
        return output
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(normalized_shape={self.normalized_shape}, "
                f"eps={self.eps}, elementwise_affine={self.elementwise_affine})")

