"""DeepSeek V3 风格的语言模型示例。

演示如何使用 tinyTorch 构建带有 MoE（混合专家）架构的 DeepSeek V3 模型。
这是一个简化的教学实现，展示核心概念。

DeepSeek V3 核心特点：
- MoE (Mixture of Experts) 架构
- Multi-head Latent Attention (MLA)
- 多查询注意力机制
- DeepSeekMoE 前馈网络

Author: TinyAI Team
"""

from tinytorch.tensor import Tensor, Shape
from tinytorch.autograd import Variable
from tinytorch.nn import Module
from tinytorch.nn.layers import Linear, LayerNorm, Embedding
from tinytorch.ml import Model
import random


class MoEGate(Module):
    """MoE 门控网络。
    
    用于选择专家和计算权重。
    """
    
    def __init__(self, hidden_size, num_experts, top_k=2):
        """初始化门控网络。
        
        Args:
            hidden_size: 隐藏层大小
            num_experts: 专家数量
            top_k: 选择的专家数量
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 门控投影
        self.gate = Linear(hidden_size, num_experts, use_bias=False)
    
    def forward(self, x: Variable):
        """前向传播。
        
        Args:
            x: 输入，形状 (batch_size, hidden_size)
        
        Returns:
            expert_indices: 选中的专家索引
            expert_weights: 专家权重
        """
        # 计算门控 logits
        gate_logits = self.gate(x)  # (batch_size, num_experts)
        
        # Softmax 归一化
        gate_weights = self._softmax(gate_logits)
        
        # Top-K 选择（简化实现：选择前 top_k 个）
        # 实际应该基于权重排序，这里简化为前 k 个
        expert_indices = list(range(self.top_k))
        expert_weights = gate_weights
        
        return expert_indices, expert_weights
    
    def _softmax(self, x: Variable) -> Variable:
        """简化的 Softmax 实现。"""
        # 对于教学目的，这里返回归一化的权重
        data = x.value.data
        
        # 计算 exp
        exp_data = [2.718 ** val for val in data]
        
        # 归一化
        total = sum(exp_data)
        if total > 0:
            normalized = [val / total for val in exp_data]
        else:
            normalized = [1.0 / len(data)] * len(data)
        
        result_tensor = Tensor(normalized, x.value.shape, 'float32')
        return Variable(result_tensor, requires_grad=x.requires_grad)


class Expert(Module):
    """单个专家网络。
    
    实现为一个简单的前馈网络。
    """
    
    def __init__(self, hidden_size, intermediate_size):
        """初始化专家。
        
        Args:
            hidden_size: 输入/输出维度
            intermediate_size: 中间层维度
        """
        super().__init__()
        self.fc1 = Linear(hidden_size, intermediate_size)
        self.fc2 = Linear(intermediate_size, hidden_size)
    
    def forward(self, x: Variable) -> Variable:
        """前向传播。"""
        # 第一层 + ReLU
        h = self.fc1(x)
        h = self._apply_relu(h)
        
        # 第二层
        output = self.fc2(h)
        return output
    
    def _apply_relu(self, x: Variable) -> Variable:
        """应用 ReLU 激活。"""
        result = x.value.relu()
        return Variable(result, requires_grad=x.requires_grad)


class DeepSeekMoE(Module):
    """DeepSeek MoE 层。
    
    使用混合专家架构的前馈网络。
    """
    
    def __init__(self, hidden_size, intermediate_size, num_experts=8, top_k=2):
        """初始化 MoE 层。
        
        Args:
            hidden_size: 隐藏层大小
            intermediate_size: 专家中间层大小
            num_experts: 专家数量
            top_k: 每次激活的专家数量
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 门控网络
        self.gate = MoEGate(hidden_size, num_experts, top_k)
        
        # 专家网络
        self.experts = []
        for _ in range(num_experts):
            expert = Expert(hidden_size, intermediate_size)
            self.experts.append(expert)
    
    def forward(self, x: Variable) -> Variable:
        """前向传播。
        
        Args:
            x: 输入，形状 (batch_size, hidden_size)
        
        Returns:
            输出，形状 (batch_size, hidden_size)
        """
        # 门控选择
        expert_indices, expert_weights = self.gate(x)
        
        # 计算专家输出（简化：使用前 top_k 个专家）
        expert_outputs = []
        for idx in expert_indices[:self.top_k]:
            if idx < len(self.experts):
                expert_out = self.experts[idx](x)
                expert_outputs.append(expert_out)
        
        # 加权组合（简化：平均）
        if expert_outputs:
            # 简单平均（实际应该使用门控权重）
            combined = expert_outputs[0]
            for expert_out in expert_outputs[1:]:
                combined_data = [
                    combined.value.data[i] + expert_out.value.data[i] 
                    for i in range(len(combined.value.data))
                ]
                combined_tensor = Tensor(combined_data, combined.value.shape, 'float32')
                combined = Variable(combined_tensor, requires_grad=combined.requires_grad)
            
            # 平均
            num_experts = len(expert_outputs)
            averaged_data = [val / num_experts for val in combined.value.data]
            result_tensor = Tensor(averaged_data, combined.value.shape, 'float32')
            return Variable(result_tensor, requires_grad=combined.requires_grad)
        
        return x


class MultiLatentAttention(Module):
    """Multi-head Latent Attention (MLA)。
    
    DeepSeek V3 的核心注意力机制。
    简化实现，展示核心思想。
    """
    
    def __init__(self, hidden_size, num_heads=8, kv_lora_rank=512):
        """初始化 MLA。
        
        Args:
            hidden_size: 隐藏层大小
            num_heads: 注意力头数
            kv_lora_rank: KV 压缩的秩
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.kv_lora_rank = kv_lora_rank
        
        # Q 投影
        self.q_proj = Linear(hidden_size, hidden_size)
        
        # KV 压缩投影（使用低秩分解）
        self.kv_a_proj = Linear(hidden_size, kv_lora_rank)
        self.kv_b_proj = Linear(kv_lora_rank, hidden_size)
        
        # 输出投影
        self.o_proj = Linear(hidden_size, hidden_size)
    
    def forward(self, x: Variable) -> Variable:
        """前向传播。
        
        Args:
            x: 输入，形状 (batch_size, hidden_size)
        
        Returns:
            输出，形状 (batch_size, hidden_size)
        """
        # Q 投影
        Q = self.q_proj(x)
        
        # KV 压缩（低秩分解）
        kv_compressed = self.kv_a_proj(x)  # 压缩到低维
        KV = self.kv_b_proj(kv_compressed)  # 恢复到原维度
        
        # 简化的注意力计算（实际应该是 scaled dot-product）
        # 这里简化为直接使用 Q 和 KV
        attention_out = Q  # 简化
        
        # 输出投影
        output = self.o_proj(attention_out)
        
        return output


class DeepSeekV3Block(Module):
    """DeepSeek V3 Transformer 块。
    
    包含 Multi-head Latent Attention 和 MoE 前馈网络。
    """
    
    def __init__(self, hidden_size=2048, num_heads=16, 
                 intermediate_size=10240, num_experts=64, top_k=6):
        """初始化 DeepSeek V3 块。
        
        Args:
            hidden_size: 隐藏层大小
            num_heads: 注意力头数
            intermediate_size: MoE 专家的中间层大小
            num_experts: 专家数量（DeepSeek V3 使用 64）
            top_k: 激活的专家数量（DeepSeek V3 使用 6）
        """
        super().__init__()
        
        # Multi-head Latent Attention
        self.attention = MultiLatentAttention(hidden_size, num_heads)
        self.norm1 = LayerNorm(hidden_size)
        
        # DeepSeek MoE
        self.moe = DeepSeekMoE(hidden_size, intermediate_size, num_experts, top_k)
        self.norm2 = LayerNorm(hidden_size)
    
    def forward(self, x: Variable) -> Variable:
        """前向传播。
        
        Args:
            x: 输入，形状 (batch_size, hidden_size)
        
        Returns:
            输出，形状 (batch_size, hidden_size)
        """
        # Attention + 残差
        attn_out = self.attention(x)
        x = self._add_residual(x, attn_out)
        x = self.norm1(x)
        
        # MoE + 残差
        moe_out = self.moe(x)
        x = self._add_residual(x, moe_out)
        x = self.norm2(x)
        
        return x
    
    def _add_residual(self, x: Variable, residual: Variable) -> Variable:
        """添加残差连接。"""
        result_data = [
            x.value.data[i] + residual.value.data[i] 
            for i in range(len(x.value.data))
        ]
        result_tensor = Tensor(result_data, x.value.shape, 'float32')
        return Variable(result_tensor, requires_grad=x.requires_grad)


class DeepSeekV3Model(Module):
    """DeepSeek V3 语言模型。
    
    简化的 DeepSeek V3 实现，用于教学演示。
    
    实际 DeepSeek V3 规格：
    - 参数量：671B
    - 层数：61
    - 隐藏层：7168
    - 注意力头：128
    - MoE 专家：256（每次激活 8 个）
    
    这里使用极小规模参数用于演示。
    """
    
    def __init__(self, vocab_size=1000, hidden_size=128, 
                 num_layers=2, num_heads=4, 
                 intermediate_size=512, num_experts=4, top_k=2):
        """初始化 DeepSeek V3 模型。
        
        Args:
            vocab_size: 词汇表大小
            hidden_size: 隐藏层大小（DeepSeek V3: 7168）
            num_layers: Transformer 层数（DeepSeek V3: 61）
            num_heads: 注意力头数（DeepSeek V3: 128）
            intermediate_size: MoE 专家中间层大小
            num_experts: 专家数量（DeepSeek V3: 256）
            top_k: 激活的专家数量（DeepSeek V3: 8）
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        print(f"\n初始化 DeepSeek V3 模型（简化版）：")
        print(f"  - 词汇表大小: {vocab_size}")
        print(f"  - 隐藏层大小: {hidden_size}")
        print(f"  - 层数: {num_layers}")
        print(f"  - 注意力头数: {num_heads}")
        print(f"  - 专家数量: {num_experts}")
        print(f"  - 激活专家数: {top_k}")
        
        # 词嵌入
        self.embedding = Embedding(vocab_size, hidden_size)
        
        # DeepSeek V3 Transformer 层
        self.layers = []
        for i in range(num_layers):
            layer = DeepSeekV3Block(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                top_k=top_k
            )
            self.layers.append(layer)
        
        # 输出层
        self.output_proj = Linear(hidden_size, vocab_size)
        
        print("✓ 模型初始化完成\n")
    
    def forward(self, input_ids: Variable) -> Variable:
        """前向传播。
        
        Args:
            input_ids: 输入 token IDs，形状 (batch_size, seq_len)
        
        Returns:
            logits，形状 (batch_size, seq_len, vocab_size)
        """
        # 词嵌入
        x = self.embedding(input_ids)  # (batch_size, seq_len, hidden_size)
        
        # 简化处理：取序列的最后一个位置
        # 实际应该处理整个序列
        batch_size = 1
        seq_len = len(x.value.data) // self.hidden_size
        last_pos_data = x.value.data[-self.hidden_size:]
        
        last_pos_tensor = Tensor(last_pos_data, Shape((batch_size, self.hidden_size)), 'float32')
        x = Variable(last_pos_tensor, requires_grad=x.requires_grad)
        
        # 通过 DeepSeek V3 层
        for i, layer in enumerate(self.layers):
            x = layer(x)
        
        # 输出投影
        logits = self.output_proj(x)  # (batch_size, vocab_size)
        
        return logits
    
    def generate(self, prompt_ids, max_new_tokens=10):
        """生成文本。
        
        Args:
            prompt_ids: 提示的 token IDs 列表
            max_new_tokens: 生成的最大 token 数
        
        Returns:
            生成的 token IDs 列表
        """
        generated = prompt_ids.copy()
        
        for _ in range(max_new_tokens):
            # 创建输入
            input_tensor = Tensor(generated, Shape((1, len(generated))), 'float32')
            input_var = Variable(input_tensor)
            
            # 前向传播
            logits = self.forward(input_var)
            
            # 贪心解码：选择概率最大的 token
            next_token = self._argmax(logits)
            
            # 添加到序列
            generated.append(next_token)
            
            # 避免生成过长
            if len(generated) > 100:
                break
        
        return generated
    
    def _argmax(self, x: Variable) -> int:
        """返回最大值的索引。"""
        data = x.value.data
        max_idx = 0
        max_val = data[0]
        
        for i, val in enumerate(data):
            if val > max_val:
                max_val = val
                max_idx = i
        
        return max_idx


def main():
    """主函数：演示 DeepSeek V3 模型。"""
    print("=" * 70)
    print("DeepSeek V3 语言模型示例")
    print("=" * 70)
    
    print("\n【说明】")
    print("DeepSeek V3 是一个使用 MoE 架构的大规模语言模型。")
    print("本示例实现了一个极简版本，用于演示核心概念：")
    print("  1. MoE (Mixture of Experts) - 混合专家架构")
    print("  2. Multi-head Latent Attention - 多头潜在注意力")
    print("  3. 门控网络 - 动态选择专家")
    print("  4. 低秩分解 - KV 缓存压缩")
    
    # 创建模型（极小规模用于演示）
    print("\n" + "=" * 70)
    print("1. 创建 DeepSeek V3 模型")
    print("=" * 70)
    
    model = DeepSeekV3Model(
        vocab_size=100,      # 实际 DeepSeek V3: 102400
        hidden_size=64,      # 实际 DeepSeek V3: 7168
        num_layers=2,        # 实际 DeepSeek V3: 61
        num_heads=4,         # 实际 DeepSeek V3: 128
        intermediate_size=256,  # 实际 DeepSeek V3: ~18432
        num_experts=4,       # 实际 DeepSeek V3: 256
        top_k=2             # 实际 DeepSeek V3: 8
    )
    
    # 测试前向传播
    print("=" * 70)
    print("2. 测试前向传播")
    print("=" * 70)
    
    # 创建输入（简单的 token 序列）
    input_ids = [1, 2, 3, 4, 5]
    print(f"\n输入 token IDs: {input_ids}")
    
    input_tensor = Tensor(input_ids, Shape((1, len(input_ids))), 'float32')
    input_var = Variable(input_tensor)
    
    # 前向传播
    print("执行前向传播...")
    logits = model.forward(input_var)
    
    print(f"✓ 输出 logits 形状: {logits.value.shape.dims}")
    print(f"  前 5 个 logits: {logits.value.data[:5]}")
    
    # 测试文本生成
    print("\n" + "=" * 70)
    print("3. 测试文本生成")
    print("=" * 70)
    
    prompt = [1, 2, 3]
    print(f"\n提示 token IDs: {prompt}")
    print("生成新的 tokens...")
    
    generated = model.generate(prompt, max_new_tokens=5)
    
    print(f"✓ 生成的完整序列: {generated}")
    print(f"  新生成的部分: {generated[len(prompt):]}")
    
    # DeepSeek V3 特性说明
    print("\n" + "=" * 70)
    print("4. DeepSeek V3 核心特性")
    print("=" * 70)
    
    print("\n【MoE 架构优势】")
    print("  ✓ 模型容量大：使用 256 个专家，总参数 671B")
    print("  ✓ 推理高效：每次只激活 8 个专家（约 37B 参数）")
    print("  ✓ 训练稳定：使用负载均衡和辅助损失")
    
    print("\n【Multi-head Latent Attention】")
    print("  ✓ KV 缓存压缩：使用低秩分解减少显存")
    print("  ✓ 计算高效：降低注意力计算复杂度")
    print("  ✓ 性能保持：保持模型表达能力")
    
    print("\n【实际应用场景】")
    print("  - 代码生成与理解")
    print("  - 数学推理")
    print("  - 多语言处理")
    print("  - 长文本建模")
    
    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)
    
    print("\n【学习建议】")
    print("1. 理解 MoE 如何通过门控网络选择专家")
    print("2. 研究低秩分解如何压缩 KV 缓存")
    print("3. 对比 DeepSeek V3 与标准 Transformer 的区别")
    print("4. 探索如何扩展到更大规模模型")
    
    print("\n【参考资料】")
    print("- DeepSeek V3 技术报告")
    print("- tinyTorch 完整文档: tutorials/")
    print("- MoE 架构详解: docs/方案.md")


if __name__ == '__main__':
    main()
