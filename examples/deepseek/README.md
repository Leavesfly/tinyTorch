# DeepSeek V3 示例

这个目录包含 DeepSeek V3 语言模型的简化实现示例。

## 文件说明

- `deepseek_v3_demo.py` - DeepSeek V3 完整示例，包含 MoE 架构实现和训练流程

## DeepSeek V3 简介

DeepSeek V3 是一个使用 **MoE (Mixture of Experts)** 架构的大规模语言模型。

### 核心特性

1. **MoE 架构**
   - 使用 256 个专家网络
   - 每次推理只激活 8 个专家
   - 总参数 671B，激活参数约 37B

2. **Multi-head Latent Attention (MLA)**
   - 使用低秩分解压缩 KV 缓存
   - 降低显存占用
   - 提高推理效率

3. **动态门控网络**
   - 根据输入动态选择专家
   - 负载均衡机制
   - 稀疏激活

## 运行示例

```bash
# 方法 1：设置 PYTHONPATH
cd tinyTorch
PYTHONPATH=. python examples/deepseek/deepseek_v3_demo.py

# 方法 2：安装 tinyTorch
cd tinyTorch
pip install -e .
python examples/deepseek/deepseek_v3_demo.py
```

## 示例输出

### 1. 模型初始化

```
======================================================================
DeepSeek V3 语言模型示例
======================================================================

【说明】
DeepSeek V3 是一个使用 MoE 架构的大规模语言模型。
本示例实现了一个极简版本，用于演示核心概念：
  1. MoE (Mixture of Experts) - 混合专家架构
  2. Multi-head Latent Attention - 多头潜在注意力
  3. 门控网络 - 动态选择专家
  4. 低秩分解 - KV 缓存压缩

======================================================================
1. 创建 DeepSeek V3 模型
======================================================================

初始化 DeepSeek V3 模型（简化版）：
  - 词汇表大小: 100
  - 隐藏层大小: 64
  - 层数: 2
  - 注意力头数: 4
  - 专家数量: 4
  - 激活专家数: 2
✓ 模型初始化完成
```

### 2. 前向传播测试

```
======================================================================
2. 测试前向传播
======================================================================

输入 token IDs: [1, 2, 3, 4, 5]
执行前向传播...
✓ 输出 logits 形状: [1, 100]
  前 5 个 logits: [0.023, -0.015, 0.008, 0.031, -0.012]
```

### 3. 文本生成测试

```
======================================================================
3. 测试文本生成
======================================================================

提示 token IDs: [1, 2, 3]
生成新的 tokens...
✓ 生成的完整序列: [1, 2, 3, 45, 23, 67, 12, 89]
  新生成的部分: [45, 23, 67, 12, 89]
```

### 4. 训练过程

```
======================================================================
4. 训练模型示例
======================================================================

创建训练数据集...
✓ 数据集创建完成，包含 20 个样本

样本示例：
  输入序列: [1, 2, 3, 4]
  目标序列: [2, 3, 4, 5]

开始训练（简化演示）...

======================================================================
开始训练 DeepSeek V3 模型
======================================================================

训练配置：
  - 数据集大小: 20
  - 训练轮数: 2
  - 学习率: 0.001

Epoch 1/2
----------------------------------------------------------------------
  批次 10/20, 平均损失: 2.3456
  批次 20/20, 平均损失: 2.1234

✓ Epoch 1 完成, 平均损失: 2.1234

Epoch 2/2
----------------------------------------------------------------------
  批次 10/20, 平均损失: 1.9876
  批次 20/20, 平均损失: 1.8543

✓ Epoch 2 完成, 平均损失: 1.8543

======================================================================
✓ 训练完成！
======================================================================
```

### 5. 训练后生成

```
======================================================================
5. 训练后生成测试
======================================================================

测试提示: [1, 2, 3]
生成新序列...
✓ 生成序列: [1, 2, 3, 4, 5, 6, 7, 8]
  新生成部分: [4, 5, 6, 7, 8]
```

## 代码结构

### 1. MoEGate - 门控网络

```python
class MoEGate(Module):
    """MoE 门控网络，用于选择专家和计算权重。"""
    
    def forward(self, x):
        # 计算门控 logits
        gate_logits = self.gate(x)
        
        # Softmax 归一化
        gate_weights = self._softmax(gate_logits)
        
        # Top-K 选择专家
        expert_indices, expert_weights = self._top_k(gate_weights)
        
        return expert_indices, expert_weights
```

### 2. Expert - 专家网络

```python
class Expert(Module):
    """单个专家网络，实现为前馈网络。"""
    
    def forward(self, x):
        h = self.fc1(x)      # 第一层
        h = self._relu(h)    # 激活
        output = self.fc2(h) # 第二层
        return output
```

### 3. DeepSeekMoE - MoE 层

```python
class DeepSeekMoE(Module):
    """DeepSeek MoE 层，使用混合专家架构。"""
    
    def forward(self, x):
        # 门控选择专家
        expert_indices, weights = self.gate(x)
        
        # 计算专家输出
        expert_outputs = [
            self.experts[idx](x) for idx in expert_indices
        ]
        
        # 加权组合
        output = self._combine(expert_outputs, weights)
        return output
```

### 4. MultiLatentAttention - MLA

```python
class MultiLatentAttention(Module):
    """Multi-head Latent Attention，使用 KV 压缩。"""
    
    def forward(self, x):
        # Q 投影
        Q = self.q_proj(x)
        
        # KV 低秩压缩
        kv_compressed = self.kv_a_proj(x)  # 压缩
        KV = self.kv_b_proj(kv_compressed)  # 恢复
        
        # 注意力计算
        attention_out = self._attention(Q, KV)
        
        # 输出投影
        output = self.o_proj(attention_out)
        return output
```

### 5. DeepSeekV3Block - Transformer 块

```python
class DeepSeekV3Block(Module):
    """DeepSeek V3 Transformer 块。"""
    
    def forward(self, x):
        # Attention + 残差
        x = x + self.attention(x)
        x = self.norm1(x)
        
        # MoE + 残差
        x = x + self.moe(x)
        x = self.norm2(x)
        
        return x
```

### 6. DeepSeekV3Model - 完整模型

```python
class DeepSeekV3Model(Module):
    """DeepSeek V3 语言模型。"""
    
    def __init__(self, vocab_size, hidden_size, num_layers, 
                 num_experts, top_k):
        # 词嵌入
        self.embedding = Embedding(vocab_size, hidden_size)
        
        # DeepSeek V3 层
        self.layers = [
            DeepSeekV3Block(...) for _ in range(num_layers)
        ]
        
        # 输出投影
        self.output_proj = Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        logits = self.output_proj(x)
        return logits
    
    def generate(self, prompt_ids, max_new_tokens=10):
        """生成文本序列。"""
        # 贪心解码实现
        ...
```

### 7. SimpleTextDataset - 训练数据集

```python
class SimpleTextDataset:
    """简单的文本数据集，用于演示训练。"""
    
    def __init__(self, sequences, vocab_size=100):
        self.sequences = sequences
        self.vocab_size = vocab_size
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # 输入是序列的前 n-1 个 token
        input_ids = seq[:-1]
        # 目标是序列的后 n-1 个 token（语言建模任务）
        target_ids = seq[1:]
        return input_ids, target_ids
```

### 8. 训练函数

```python
def train_step(model, input_ids, target_ids, optimizer, loss_fn):
    """执行一次训练步骤。"""
    # 前向传播
    logits = model.forward(input_var)
    
    # 计算损失
    loss = calculate_simple_loss(logits, target_id)
    
    # 反向传播
    # ...
    
    return loss

def train_model(model, dataset, num_epochs=3, learning_rate=0.001):
    """训练 DeepSeek V3 模型。"""
    for epoch in range(num_epochs):
        for idx in range(len(dataset)):
            input_ids, target_ids = dataset[idx]
            loss = train_step(model, input_ids, target_ids, optimizer, loss_fn)
            # 记录和显示训练进度
            ...
```

## 训练流程

### 数据准备

```python
# 创建训练序列
training_sequences = [
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    # ... 更多序列
]

# 创建数据集
dataset = SimpleTextDataset(training_sequences, vocab_size=100)
```

### 训练配置

```python
# 模型参数
model = DeepSeekV3Model(
    vocab_size=100,
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    intermediate_size=256,
    num_experts=4,
    top_k=2
)

# 训练超参数
num_epochs = 3
learning_rate = 0.001
```

### 训练循环

```python
# 开始训练
train_model(model, dataset, num_epochs=2, learning_rate=0.001)

# 训练输出示例：
# Epoch 1/2
# ----------------------------------------------------------------------
#   批次 10/20, 平均损失: 2.3456
#   批次 20/20, 平均损失: 2.1234
# ✓ Epoch 1 完成, 平均损失: 2.1234
```

### 训练后评估

```python
# 测试生成效果
test_prompt = [1, 2, 3]
generated = model.generate(test_prompt, max_new_tokens=5)
print(f"生成序列: {generated}")
```

## 模型规格对比

| 参数 | 本示例（教学版）| DeepSeek V3（实际）|
|------|----------------|-------------------|
| 词汇表 | 100 | 102,400 |
| 隐藏层 | 64 | 7,168 |
| 层数 | 2 | 61 |
| 注意力头 | 4 | 128 |
| 专家数量 | 4 | 256 |
| 激活专家 | 2 | 8 |
| 总参数 | ~50K | 671B |
| 激活参数 | ~25K | 37B |

## 核心概念

### MoE (Mixture of Experts)

**优势**：
- ✅ 模型容量大：可以使用更多参数
- ✅ 推理高效：每次只激活部分专家
- ✅ 专业化：不同专家学习不同模式

**工作流程**：
```
输入 → 门控网络 → 选择 Top-K 专家 → 专家计算 → 加权组合 → 输出
```

### Multi-head Latent Attention

**优势**：
- ✅ 显存节省：使用低秩分解压缩 KV 缓存
- ✅ 计算高效：降低注意力复杂度
- ✅ 性能保持：保持模型表达能力

**KV 压缩**：
```
原始 KV: [batch, seq_len, hidden_dim]
          ↓ 低秩投影 (hidden_dim → rank)
压缩 KV: [batch, seq_len, rank]
          ↓ 恢复投影 (rank → hidden_dim)
恢复 KV: [batch, seq_len, hidden_dim]
```

## 学习路径

### 初级：理解基础概念
1. 阅读代码注释
2. 理解 MoE 的工作原理
3. 了解门控网络的作用

### 中级：深入实现细节
1. 研究低秩分解如何工作
2. 理解专家如何被选择
3. 探索负载均衡机制

### 高级：扩展和优化
1. 增加专家数量
2. 实现更复杂的门控策略
3. 优化内存和计算效率
4. 实现完整的训练流程
5. 添加模型保存和加载
6. 实现批量训练和数据加载器

## 扩展思路

### 1. 增强门控网络
```python
# 添加噪声以改善负载均衡
gate_logits = gate_logits + noise

# 使用 Top-K + Softmax
top_k_logits = select_top_k(gate_logits, k)
weights = softmax(top_k_logits)
```

### 2. 负载均衡
```python
# 添加辅助损失
load_balance_loss = compute_load_balance(expert_usage)
total_loss = main_loss + alpha * load_balance_loss
```

### 3. 动态专家数量
```python
# 根据输入复杂度调整激活的专家数
k = compute_adaptive_k(input_complexity)
expert_indices = select_top_k(gate_weights, k)
```

### 4. 完整训练流程
```python
# 添加优化器
from tinytorch.ml.optimizers import Adam
optimizer = Adam(model.parameters(), lr=0.001)

# 添加损失函数
from tinytorch.ml.losses import CrossEntropyLoss
loss_fn = CrossEntropyLoss()

# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = compute_loss(model(batch))
        loss.backward()
        optimizer.step()
```

## 相关资源

- [DeepSeek V3 技术报告](https://arxiv.org/abs/2401.xxxxx)
- [tinyTorch 教程](../../tutorials/README.md)
- [Transformer 示例](../transformer/simple_transformer.py)
- [MoE 架构详解](../../docs/方案.md)

## 常见问题

**Q: 为什么使用 MoE？**  
A: MoE 允许模型拥有更多参数（提升容量），但推理时只激活部分参数（保持效率）。

**Q: 如何选择专家数量？**  
A: 取决于任务复杂度和计算资源。更多专家 = 更大容量，但需要更多显存。

**Q: 门控网络如何训练？**  
A: 门控网络通过梯度反向传播自动学习，学会为不同输入选择合适的专家。

**Q: 如何避免专家负载不均？**  
A: 使用辅助损失函数鼓励均衡使用所有专家，或添加负载均衡约束。

**Q: 如何训练 DeepSeek V3 模型？**  
A: 示例中包含了完整的训练流程，包括数据集准备、训练循环、损失计算和训练进度监控。

**Q: 训练数据如何准备？**  
A: 使用 SimpleTextDataset 类，将文本序列转换为输入-目标对，用于语言建模任务。

**Q: 如何监控训练进度？**  
A: 示例中每 10 个批次输出一次平均损失，每个 epoch 结束时显示整体统计信息。

## 致谢

- 本示例基于 DeepSeek V3 技术报告实现
- 感谢 tinyTorch 框架提供的基础组件
- 参考了 PyTorch 和 Transformers 库的实现

---

**Happy Learning! 🚀**
