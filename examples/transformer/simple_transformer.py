"""使用 Transformer 注意力机制的示例。

演示如何使用 tinyTorch 构建简单的 Transformer 模型。

Author: TinyAI Team
"""

from tinytorch.tensor import Tensor
from tinytorch.autograd import Variable
from tinytorch.nn import Module
from tinytorch.nn.layers import MultiHeadAttention, Linear, LayerNorm, Dropout, Embedding
from tinytorch.ml import Model, Monitor
import random


class TransformerBlock(Module):
    """Transformer 编码器块。
    
    包含多头自注意力层和前馈网络。
    
    结构：
        LayerNorm -> MultiHeadAttention -> Residual
        LayerNorm -> FFN -> Residual
    """
    
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, dropout=0.1):
        """初始化 Transformer 块。
        
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            ff_dim: 前馈网络维度
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 多头注意力
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = LayerNorm(embed_dim)
        self.dropout1 = Dropout(dropout)
        
        # 前馈网络
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = LayerNorm(embed_dim)
        self.dropout2 = Dropout(dropout)
    
    def forward(self, x: Variable) -> Variable:
        """前向传播。
        
        Args:
            x: 输入序列，形状 (batch_size, seq_len, embed_dim)
        
        Returns:
            输出序列，形状 (batch_size, seq_len, embed_dim)
        """
        # 多头自注意力 + 残差
        attn_out = self.attention(x)
        attn_out = self.dropout1(attn_out)
        x = self._add_residual(x, attn_out)
        x = self.norm1(x)
        
        # 前馈网络 + 残差
        ffn_out = self.ffn(x)
        ffn_out = self.dropout2(ffn_out)
        x = self._add_residual(x, ffn_out)
        x = self.norm2(x)
        
        return x
    
    def _add_residual(self, x: Variable, residual: Variable) -> Variable:
        """添加残差连接。"""
        result = x.value.add(residual.value)
        return Variable(result, requires_grad=x.requires_grad)


class FeedForward(Module):
    """前馈网络。
    
    结构：Linear -> ReLU -> Dropout -> Linear
    """
    
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        """初始化前馈网络。
        
        Args:
            embed_dim: 输入/输出维度
            ff_dim: 隐藏层维度
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.fc1 = Linear(embed_dim, ff_dim)
        self.dropout = Dropout(dropout)
        self.fc2 = Linear(ff_dim, embed_dim)
    
    def forward(self, x: Variable) -> Variable:
        """前向传播。"""
        batch_size, seq_len, embed_dim = x.value.shape.dims
        
        # 展平处理
        flat_data = x.value.data
        flat_tensor = Tensor(flat_data, (batch_size * seq_len, embed_dim), 'float32')
        flat_var = Variable(flat_tensor, requires_grad=x.requires_grad)
        
        # 第一层：Linear + ReLU
        h = self.fc1(flat_var)
        h = self._apply_relu(h)
        h = self.dropout(h)
        
        # 第二层：Linear
        output = self.fc2(h)
        
        # 重塑回 (batch_size, seq_len, embed_dim)
        output_data = output.value.data
        output_tensor = Tensor(output_data, (batch_size, seq_len, embed_dim), 'float32')
        
        return Variable(output_tensor, requires_grad=output.requires_grad)
    
    def _apply_relu(self, x: Variable) -> Variable:
        """应用 ReLU 激活。"""
        result = x.value.relu()
        return Variable(result, requires_grad=x.requires_grad)


class SimpleTransformer(Module):
    """简单的 Transformer 模型。
    
    用于序列到序列的任务。
    """
    
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, 
                 num_layers=2, ff_dim=1024, max_seq_len=100):
        """初始化 Transformer。
        
        Args:
            vocab_size: 词汇表大小
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            num_layers: Transformer 层数
            ff_dim: 前馈网络维度
            max_seq_len: 最大序列长度
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # 词嵌入
        self.token_embedding = Embedding(vocab_size, embed_dim)
        
        # Transformer 层
        self.layers = []
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, ff_dim)
            self.layers.append(layer)
        
        # 输出层
        self.output_proj = Linear(embed_dim, vocab_size)
    
    def forward(self, x: Variable) -> Variable:
        """前向传播。
        
        Args:
            x: 输入序列，形状 (batch_size, seq_len)
        
        Returns:
            输出 logits，形状 (batch_size, seq_len, vocab_size)
        """
        # 词嵌入
        x = self.token_embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # 通过 Transformer 层
        for layer in self.layers:
            x = layer(x)
        
        # 投影到词汇表
        batch_size, seq_len, embed_dim = x.value.shape.dims
        
        # 展平处理
        flat_data = x.value.data
        flat_tensor = Tensor(flat_data, (batch_size * seq_len, embed_dim), 'float32')
        flat_var = Variable(flat_tensor, requires_grad=x.requires_grad)
        
        # 输出投影
        output = self.output_proj(flat_var)  # (batch_size * seq_len, vocab_size)
        
        # 重塑
        output_data = output.value.data
        output_tensor = Tensor(output_data, (batch_size, seq_len, self.vocab_size), 'float32')
        
        return Variable(output_tensor, requires_grad=output.requires_grad)


def main():
    """主函数。"""
    print("=" * 60)
    print("tinyTorch 框架 - Transformer 注意力机制示例")
    print("=" * 60)
    
    # 设置参数
    vocab_size = 100
    embed_dim = 128
    num_heads = 4
    num_layers = 2
    ff_dim = 512
    seq_len = 10
    batch_size = 2
    
    print("\n步骤 1: 模型参数")
    print(f"  词汇表大小: {vocab_size}")
    print(f"  嵌入维度: {embed_dim}")
    print(f"  注意力头数: {num_heads}")
    print(f"  Transformer 层数: {num_layers}")
    print(f"  前馈网络维度: {ff_dim}")
    
    print("\n步骤 2: 构建 Transformer 模型")
    transformer = SimpleTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim
    )
    model = Model(name='SimpleTransformer', module=transformer)
    print(f"  模型: {model.name}")
    print(f"  参数数量: {len(model.parameters())}")
    
    print("\n步骤 3: 生成测试输入")
    # 生成随机序列
    test_seq = [[random.randint(0, vocab_size - 1) for _ in range(seq_len)] 
                for _ in range(batch_size)]
    
    print(f"  批次大小: {batch_size}")
    print(f"  序列长度: {seq_len}")
    print(f"  示例序列: {test_seq[0][:5]}...")
    
    # 创建输入张量
    from tinytorch.tensor import Tensor, Shape
    input_data = []
    for seq in test_seq:
        input_data.extend([float(x) for x in seq])
    
    input_tensor = Tensor(input_data, Shape((batch_size, seq_len)), 'float32')
    input_var = Variable(input_tensor, requires_grad=False)
    
    print("\n步骤 4: 前向传播")
    print("注意: 这是一个简化的演示，展示 Transformer 的基本结构")
    
    monitor = Monitor()
    monitor.start()
    
    # 前向传播
    print("  执行前向传播...")
    output = transformer(input_var)
    
    print(f"  输出形状: {output.value.shape.dims}")
    print(f"  预期形状: (batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size})")
    
    # 提取第一个样本第一个位置的预测
    first_logits = []
    for i in range(vocab_size):
        first_logits.append(output.value.data[i])
    
    predicted_token = first_logits.index(max(first_logits))
    print(f"\n  第一个位置的预测 token: {predicted_token}")
    print(f"  概率分布（前5个）: {[f'{x:.4f}' for x in first_logits[:5]]}")
    
    print("\n步骤 5: 注意力机制说明")
    print("  Transformer 核心组件:")
    print("    1. MultiHeadAttention - 多头自注意力")
    print("       - 将输入投影为 Query、Key、Value")
    print("       - 计算注意力权重：softmax(Q @ K.T / sqrt(d_k))")
    print("       - 加权求和：Attention @ V")
    print("    2. FeedForward - 前馈网络")
    print("       - 两层全连接网络")
    print("       - 使用 ReLU 激活函数")
    print("    3. LayerNorm - 层归一化")
    print("    4. Residual Connection - 残差连接")
    
    print("\n" + "=" * 60)
    print("Transformer 注意力机制示例完成!")
    print("=" * 60)
    
    print("\n说明:")
    print("  tinyTorch 已实现 Transformer 的核心组件：")
    print("  ✓ MultiHeadAttention（多头注意力）")
    print("  ✓ LayerNorm（层归一化）")
    print("  ✓ Dropout（正则化）")
    print("  ✓ Linear（全连接层）")
    print("  ✓ Embedding（词嵌入）")
    print("\n  这些组件可以组合构建完整的 Transformer 模型！")


if __name__ == '__main__':
    main()
