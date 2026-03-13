"""层归一化（LayerNorm）、Dropout 和 Embedding 层实现。

Author: TinyAI Team
"""

from tinytorch.nn.module import Module
from tinytorch.nn.parameter import Parameter
from tinytorch.autograd.ops.nn import EmbeddingLookup as _EmbeddingLookup
from tinytorch.autograd.tensor import Tensor
from tinytorch.ndarr.ndarray import NdArray
from tinytorch.nn import init
from tinytorch.utils import random as tt_random


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
        # 评估模式下不做任何操作
        if not self.training:
            return input
        
        # 训练模式下应用 dropout
        if self.p == 0:
            return input
        
        # 生成 dropout mask
        mask_data = []
        for _ in range(len(input.value.data)):
            if tt_random.random() < self.p:
                mask_data.append(0.0)
            else:
                # 缩放以保持期望值不变
                mask_data.append(1.0 / (1.0 - self.p))
        
        mask = NdArray(mask_data, input.value.shape, input.value.dtype)
        mask_var = Tensor(mask, requires_grad=False)
        
        return input * mask_var
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class Embedding(Module):
    """词嵌入层。
    
    将整数索引映射到稠密向量。常用于 NLP 任务中的词嵌入。
    
    Attributes:
        num_embeddings: 词汇表大小
        embedding_dim: 嵌入维度
        weight: 嵌入矩阵，形状为 (num_embeddings, embedding_dim)
    
    Example:
        >>> embedding = Embedding(num_embeddings=1000, embedding_dim=128)
        >>> # 输入是词索引
        >>> indices = Tensor(NdArray([[1, 2, 3], [4, 5, 6]]))
        >>> embedded = embedding(indices)
        >>> # 输出形状: (2, 3, 128)
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 padding_idx: int = None, name: str = None):
        """初始化 Embedding 层。
        
        Args:
            num_embeddings: 词汇表大小
            embedding_dim: 嵌入向量的维度
            padding_idx: 如果指定，该索引对应的嵌入向量将被初始化为零
            name: 层的名称
        """
        super().__init__(name=name or 'Embedding')
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # 初始化嵌入矩阵 (num_embeddings, embedding_dim)
        self.weight = Parameter(
            NdArray.randn((num_embeddings, embedding_dim)),
            name=f'{self.name}.weight'
        )
        
        # 如果指定了 padding_idx，将其初始化为零
        if padding_idx is not None:
            if not 0 <= padding_idx < num_embeddings:
                raise ValueError(f"padding_idx must be in [0, {num_embeddings})")
            # 将 padding_idx 对应的行置零
            for j in range(embedding_dim):
                self.weight.value.data[padding_idx * embedding_dim + j] = 0.0
    
    def forward(self, input: Tensor) -> Tensor:
        """前向传播。
        
        Args:
            input: 输入变量，包含整数索引，形状为 (batch_size, seq_len) 或 (batch_size,)
        
        Returns:
            嵌入向量，形状为 (*input.shape, embedding_dim)
        """
        return _EmbeddingLookup(self.num_embeddings, self.embedding_dim, self.padding_idx)(
            input, self.weight
        )
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(num_embeddings={self.num_embeddings}, "
                f"embedding_dim={self.embedding_dim})")


