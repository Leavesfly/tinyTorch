"""注意力机制层。

Author: TinyAI Team
"""

import math
from tinytorch.nn.module import Module
from tinytorch.nn.parameter import Parameter
from tinytorch.nn.layers.linear import Linear
from tinytorch.autograd import Variable
from tinytorch.tensor import Tensor, Shape


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
        >>> x = Variable(Tensor.randn((batch_size, seq_len, 512)))
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
    
    def forward(self, query: Variable, key: Variable = None, value: Variable = None,
                mask: Variable = None) -> Variable:
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
        Q = self._split_heads(Q, batch_size, seq_len)
        K = self._split_heads(K, batch_size, seq_len)
        V = self._split_heads(V, batch_size, seq_len)
        
        # 计算注意力分数：Q @ K.T / sqrt(d_k)
        attn_scores = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        output = self._merge_heads(attn_scores, batch_size, seq_len)
        
        # 输出投影
        output = self.W_o(output)
        
        return output
    
    def _split_heads(self, x: Variable, batch_size: int, seq_len: int) -> Variable:
        """将最后一个维度分割为多头。
        
        Args:
            x: 输入张量，形状 (batch_size, seq_len, embed_dim)
            batch_size: 批次大小
            seq_len: 序列长度
        
        Returns:
            重塑后的张量，形状 (batch_size, num_heads, seq_len, head_dim)
        """
        # 简化实现：将 (batch_size, seq_len, embed_dim) 重塑为
        # (batch_size, seq_len, num_heads, head_dim) 再转置
        
        # 重塑数据
        data = []
        for b in range(batch_size):
            for h in range(self.num_heads):
                for s in range(seq_len):
                    for d in range(self.head_dim):
                        # 原始索引：(b, s, h * head_dim + d)
                        orig_idx = (b * seq_len * self.embed_dim +
                                   s * self.embed_dim +
                                   h * self.head_dim + d)
                        data.append(x.value.data[orig_idx])
        
        new_shape = Shape((batch_size, self.num_heads, seq_len, self.head_dim))
        new_tensor = Tensor(data, new_shape, 'float32')
        return Variable(new_tensor, requires_grad=x.requires_grad)
    
    def _merge_heads(self, x: Variable, batch_size: int, seq_len: int) -> Variable:
        """合并多头。
        
        Args:
            x: 输入张量，形状 (batch_size, num_heads, seq_len, head_dim)
            batch_size: 批次大小
            seq_len: 序列长度
        
        Returns:
            合并后的张量，形状 (batch_size, seq_len, embed_dim)
        """
        data = []
        for b in range(batch_size):
            for s in range(seq_len):
                for h in range(self.num_heads):
                    for d in range(self.head_dim):
                        # 原始索引：(b, h, s, d)
                        orig_idx = (b * self.num_heads * seq_len * self.head_dim +
                                   h * seq_len * self.head_dim +
                                   s * self.head_dim +
                                   d)
                        data.append(x.value.data[orig_idx])
        
        new_shape = Shape((batch_size, seq_len, self.embed_dim))
        new_tensor = Tensor(data, new_shape, 'float32')
        return Variable(new_tensor, requires_grad=x.requires_grad)
    
    def _scaled_dot_product_attention(self, Q: Variable, K: Variable, 
                                     V: Variable, mask: Variable = None) -> Variable:
        """缩放点积注意力。
        
        Args:
            Q: Query，形状 (batch_size, num_heads, seq_len, head_dim)
            K: Key，形状 (batch_size, num_heads, seq_len, head_dim)
            V: Value，形状 (batch_size, num_heads, seq_len, head_dim)
            mask: 掩码（可选）
        
        Returns:
            注意力输出，形状 (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = Q.value.shape.dims
        
        # 计算 Q @ K.T
        # 简化实现：对每个 (batch, head) 对进行矩阵乘法
        scores_data = []
        
        for b in range(batch_size):
            for h in range(num_heads):
                # 提取 Q[b, h] 和 K[b, h]
                q_matrix = self._extract_matrix(Q, b, h, seq_len, head_dim)
                k_matrix = self._extract_matrix(K, b, h, seq_len, head_dim)
                
                # Q @ K.T，结果形状 (seq_len, seq_len)
                for i in range(seq_len):
                    for j in range(seq_len):
                        score = 0.0
                        for d in range(head_dim):
                            score += q_matrix[i * head_dim + d] * k_matrix[j * head_dim + d]
                        # 缩放
                        score /= math.sqrt(head_dim)
                        scores_data.append(score)
        
        scores_shape = Shape((batch_size, num_heads, seq_len, seq_len))
        scores_tensor = Tensor(scores_data, scores_shape, 'float32')
        scores = Variable(scores_tensor, requires_grad=Q.requires_grad)
        
        # Softmax
        attn_weights = self._softmax(scores, dim=-1)
        
        # attn_weights @ V
        output_data = []
        for b in range(batch_size):
            for h in range(num_heads):
                # 提取注意力权重和 V
                attn_matrix = self._extract_attn_matrix(attn_weights, b, h, seq_len)
                v_matrix = self._extract_matrix(V, b, h, seq_len, head_dim)
                
                # 矩阵乘法：(seq_len, seq_len) @ (seq_len, head_dim)
                for i in range(seq_len):
                    for d in range(head_dim):
                        val = 0.0
                        for j in range(seq_len):
                            val += attn_matrix[i * seq_len + j] * v_matrix[j * head_dim + d]
                        output_data.append(val)
        
        output_shape = Shape((batch_size, num_heads, seq_len, head_dim))
        output_tensor = Tensor(output_data, output_shape, 'float32')
        return Variable(output_tensor, requires_grad=Q.requires_grad)
    
    def _extract_matrix(self, x: Variable, b: int, h: int, 
                       seq_len: int, head_dim: int) -> list:
        """提取 (b, h) 位置的矩阵。"""
        data = []
        for s in range(seq_len):
            for d in range(head_dim):
                idx = (b * self.num_heads * seq_len * head_dim +
                      h * seq_len * head_dim +
                      s * head_dim +
                      d)
                data.append(x.value.data[idx])
        return data
    
    def _extract_attn_matrix(self, x: Variable, b: int, h: int, seq_len: int) -> list:
        """提取注意力权重矩阵。"""
        data = []
        for i in range(seq_len):
            for j in range(seq_len):
                idx = (b * self.num_heads * seq_len * seq_len +
                      h * seq_len * seq_len +
                      i * seq_len +
                      j)
                data.append(x.value.data[idx])
        return data
    
    def _softmax(self, x: Variable, dim: int = -1) -> Variable:
        """Softmax 操作（沿最后一个维度）。"""
        batch_size, num_heads, seq_len_q, seq_len_k = x.value.shape.dims
        
        # 对每一行应用 softmax
        result_data = []
        for b in range(batch_size):
            for h in range(num_heads):
                for i in range(seq_len_q):
                    # 提取一行
                    row = []
                    for j in range(seq_len_k):
                        idx = (b * num_heads * seq_len_q * seq_len_k +
                              h * seq_len_q * seq_len_k +
                              i * seq_len_k +
                              j)
                        row.append(x.value.data[idx])
                    
                    # 计算 softmax
                    max_val = max(row)
                    exp_row = [math.exp(val - max_val) for val in row]
                    sum_exp = sum(exp_row)
                    softmax_row = [val / sum_exp for val in exp_row]
                    
                    result_data.extend(softmax_row)
        
        result_tensor = Tensor(result_data, x.value.shape, 'float32')
        return Variable(result_tensor, requires_grad=x.requires_grad)
    
    def __repr__(self) -> str:
        """返回层的字符串表示。"""
        return (f"MultiHeadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
                f"head_dim={self.head_dim})")
