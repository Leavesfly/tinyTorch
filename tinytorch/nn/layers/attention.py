"""注意力机制层。

Author: TinyAI Team
"""

from tinytorch.nn.module import Module
from tinytorch.nn.layers.linear import Linear
from tinytorch.autograd import Tensor
from tinytorch.autograd.ops.nn import MergeHeads as _MergeHeads
from tinytorch.autograd.ops.nn import ScaledDotProductAttention as _ScaledDotProductAttention
from tinytorch.autograd.ops.nn import SplitHeads as _SplitHeads


class MultiHeadAttention(Module):
    """多头注意力机制。
    
    实现 Transformer 中的多头自注意力机制。
    
    公式：
        Q = X @ W_q, K = X @ W_k, V = X @ W_v
        Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
        MultiHead = Concat(head_1, ..., head_h) @ W_o
    
    Attributes:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        head_dim: 每个头的维度
        W_q: Query 投影权重
        W_k: Key 投影权重
        W_v: Value 投影权重
        W_o: 输出投影权重
    
    Example:
        >>> attn = MultiHeadAttention(embed_dim=512, num_heads=8)
        >>> x = Tensor(NdArray.randn((batch_size, seq_len, 512)))
        >>> output = attn(x)
        >>> print(output.value.shape)
        (batch_size, seq_len, 512)
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """初始化多头注意力层。
        
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout 概率（暂不实现）
        """
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        # Q、K、V 投影层
        self.W_q = Linear(embed_dim, embed_dim, use_bias=True)
        self.W_k = Linear(embed_dim, embed_dim, use_bias=True)
        self.W_v = Linear(embed_dim, embed_dim, use_bias=True)
        
        # 输出投影层
        self.W_o = Linear(embed_dim, embed_dim, use_bias=True)
    
    def forward(self, query: Tensor, key: Tensor = None, value: Tensor = None,
                mask: Tensor = None) -> Tensor:
        """前向传播。
        
        Args:
            query: Query 张量，形状 (batch_size, seq_len, embed_dim)
            key: Key 张量，默认为 query（自注意力）
            value: Value 张量，默认为 query
            mask: 注意力掩码（可选）
        
        Returns:
            输出张量，形状 (batch_size, seq_len, embed_dim)
        """
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size, seq_len, embed_dim = query.value.shape.dims
        
        # 线性投影
        Q = self.W_q(query)  # (batch_size, seq_len, embed_dim)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 重塑为多头：(batch_size, num_heads, seq_len, head_dim)
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # 计算注意力分数：Q @ K.T / sqrt(d_k)
        attn_scores = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        output = self._merge_heads(attn_scores)
        
        # 输出投影
        output = self.W_o(output)
        
        return output
    
    def _split_heads(self, x: Tensor) -> Tensor:
        """将最后一个维度分割为多头。
        
        Args:
            x: 输入张量，形状 (batch_size, seq_len, embed_dim)
        Returns:
            重塑后的张量，形状 (batch_size, num_heads, seq_len, head_dim)
        """
        return _SplitHeads(self.num_heads, self.head_dim, self.embed_dim)(x)
    
    def _merge_heads(self, x: Tensor) -> Tensor:
        """合并多头。
        
        Args:
            x: 输入张量，形状 (batch_size, num_heads, seq_len, head_dim)
        Returns:
            合并后的张量，形状 (batch_size, seq_len, embed_dim)
        """
        return _MergeHeads(self.num_heads, self.head_dim, self.embed_dim)(x)
    
    def _scaled_dot_product_attention(self, Q: Tensor, K: Tensor,
                                      V: Tensor, mask: Tensor = None) -> Tensor:
        """缩放点积注意力。
        
        Args:
            Q: Query，形状 (batch_size, num_heads, seq_len, head_dim)
            K: Key，形状 (batch_size, num_heads, seq_len, head_dim)
            V: Value，形状 (batch_size, num_heads, seq_len, head_dim)
            mask: 掩码（可选）
        
        Returns:
            注意力输出，形状 (batch_size, num_heads, seq_len, head_dim)
        """
        op = _ScaledDotProductAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=self.dropout,
            training=self.training,
        )
        if mask is not None:
            return op(Q, K, V, mask)
        return op(Q, K, V)

    def __repr__(self) -> str:
        """返回层的字符串表示。"""
        return (f"MultiHeadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
                f"head_dim={self.head_dim})")
