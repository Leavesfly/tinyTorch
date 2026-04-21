"""词嵌入层实现。

Author: TinyAI Team
"""

from tinytorch.nn.module import Module
from tinytorch.nn.parameter import Parameter
from tinytorch.autograd.ops.nn import EmbeddingLookup as _EmbeddingLookup
from tinytorch.autograd.tensor import Tensor
from tinytorch.ndarr.ndarray import NdArray


class Embedding(Module):
    """词嵌入层。

    将整数索引映射到稠密向量。常用于 NLP 任务中的词嵌入。

    Attributes:
        num_embeddings: 词汇表大小
        embedding_dim: 嵌入维度
        weight: 嵌入矩阵，形状为 (num_embeddings, embedding_dim)

    Example:
        >>> embedding = Embedding(num_embeddings=1000, embedding_dim=128)
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

        self.weight = Parameter(
            NdArray.randn((num_embeddings, embedding_dim)),
            name=f'{self.name}.weight'
        )

        if padding_idx is not None:
            if not 0 <= padding_idx < num_embeddings:
                raise ValueError(f"padding_idx must be in [0, {num_embeddings})")
            for col in range(embedding_dim):
                self.weight.value.data[padding_idx * embedding_dim + col] = 0.0

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
