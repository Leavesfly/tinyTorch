"""层归一化（LayerNorm）、Dropout 和 Embedding 层实现。

Author: TinyAI Team
"""

from tinytorch.nn.module import Module
from tinytorch.nn.parameter import Parameter
from tinytorch.autograd.variable import Variable
from tinytorch.tensor.tensor import Tensor
from tinytorch.tensor.shape import Shape
from tinytorch.nn import init
import random


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
        >>> x = Variable(Tensor.randn((32, 10)))
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
            self.weight = Parameter(Tensor.ones(normalized_shape), 
                                   name=f'{self.name}.weight')
            # beta 参数（偏移）
            self.bias = Parameter(Tensor.zeros(normalized_shape), 
                                 name=f'{self.name}.bias')
        else:
            self.weight = None
            self.bias = None
    
    def forward(self, input: Variable) -> Variable:
        """前向传播。
        
        Args:
            input: 输入变量
        
        Returns:
            归一化后的输出变量
        """
        # 计算均值
        mean = input.mean()
        
        # 计算方差: var = mean((x - mean)^2)
        diff = input - mean
        variance = (diff * diff).mean()
        
        # 归一化: (x - mean) / sqrt(var + eps)
        std = (variance + Variable(Tensor([self.eps]), requires_grad=False)).sqrt()
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
        >>> x = Variable(Tensor.randn((32, 10)))
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
    
    def forward(self, input: Variable) -> Variable:
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
            if random.random() < self.p:
                mask_data.append(0.0)
            else:
                # 缩放以保持期望值不变
                mask_data.append(1.0 / (1.0 - self.p))
        
        mask = Tensor(mask_data, input.value.shape, input.value.dtype)
        mask_var = Variable(mask, requires_grad=False)
        
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
        >>> indices = Variable(Tensor([[1, 2, 3], [4, 5, 6]]))
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
            Tensor.randn((num_embeddings, embedding_dim)),
            name=f'{self.name}.weight'
        )
        
        # 如果指定了 padding_idx，将其初始化为零
        if padding_idx is not None:
            if not 0 <= padding_idx < num_embeddings:
                raise ValueError(f"padding_idx must be in [0, {num_embeddings})")
            # 将 padding_idx 对应的行置零
            for j in range(embedding_dim):
                self.weight.value.data[padding_idx * embedding_dim + j] = 0.0
    
    def forward(self, input: Variable) -> Variable:
        """前向传播。
        
        Args:
            input: 输入变量，包含整数索引，形状为 (batch_size, seq_len) 或 (batch_size,)
        
        Returns:
            嵌入向量，形状为 (*input.shape, embedding_dim)
        """
        # 获取输入索引
        input_shape = input.value.shape.dims
        input_data = input.value.data
        
        # 计算输出形状
        output_shape_tuple = input_shape + (self.embedding_dim,)
        output_shape = Shape(output_shape_tuple)
        output_size = 1
        for dim in output_shape_tuple:
            output_size *= dim
        
        # 执行查表操作
        output_data = []
        for idx_val in input_data:
            idx = int(idx_val)
            if not 0 <= idx < self.num_embeddings:
                raise IndexError(f"Index {idx} out of range [0, {self.num_embeddings})")
            
            # 提取对应的嵌入向量
            start = idx * self.embedding_dim
            end = start + self.embedding_dim
            output_data.extend(self.weight.value.data[start:end])
        
        output_tensor = Tensor(output_data, output_shape, 'float32')
        return Variable(output_tensor, requires_grad=input.requires_grad)
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(num_embeddings={self.num_embeddings}, "
                f"embedding_dim={self.embedding_dim})")
