"""使用 RNN/LSTM 进行序列建模的示例。

演示如何使用 tinyTorch 构建循环神经网络进行序列预测。

Author: TinyAI Team
"""

from tinytorch.tensor import Tensor
from tinytorch.autograd import Variable
from tinytorch.nn import Module
from tinytorch.nn.layers import LSTM, Linear, Embedding, Dropout
from tinytorch.ml import Model, DataSet, Adam, CrossEntropyLoss, Monitor
import random


class LSTMLanguageModel(Module):
    """基于 LSTM 的语言模型。
    
    网络结构：
        Embedding(vocab_size, embed_dim) ->
        LSTM(embed_dim, hidden_size) ->
        Linear(hidden_size, vocab_size)
    """
    
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128):
        """初始化语言模型。
        
        Args:
            vocab_size: 词汇表大小
            embed_dim: 词嵌入维度
            hidden_size: LSTM 隐藏层大小
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        
        # 词嵌入层
        self.embedding = Embedding(vocab_size, embed_dim)
        
        # LSTM 层
        self.lstm = LSTM(input_size=embed_dim, hidden_size=hidden_size)
        
        # 输出层
        self.fc = Linear(hidden_size, vocab_size)
        
        # Dropout（可选）
        self.dropout = Dropout(p=0.1)
    
    def forward(self, x: Variable) -> Variable:
        """前向传播。
        
        Args:
            x: 输入序列，形状 (batch_size, seq_len)，元素为词索引
        
        Returns:
            输出 logits，形状 (batch_size, seq_len, vocab_size)
        """
        # 词嵌入
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_size)
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # 投影到词汇表
        # 需要对每个时间步应用 Linear
        batch_size, seq_len, hidden_size = lstm_out.value.shape.dims
        
        # 简化实现：展平处理
        flat_data = lstm_out.value.data
        flat_tensor = Tensor(flat_data, (batch_size * seq_len, hidden_size), 'float32')
        flat_var = Variable(flat_tensor, requires_grad=lstm_out.requires_grad)
        
        # 通过全连接层
        output = self.fc(flat_var)  # (batch_size * seq_len, vocab_size)
        
        # 重塑回 (batch_size, seq_len, vocab_size)
        output_data = output.value.data
        output_tensor = Tensor(output_data, (batch_size, seq_len, self.vocab_size), 'float32')
        
        return Variable(output_tensor, requires_grad=output.requires_grad)


def generate_toy_sequences(vocab_size=50, seq_len=10, num_samples=100):
    """生成玩具序列数据。
    
    Args:
        vocab_size: 词汇表大小
        seq_len: 序列长度
        num_samples: 样本数量
    
    Returns:
        (input_seqs, target_seqs) 元组
    """
    input_seqs = []
    target_seqs = []
    
    for _ in range(num_samples):
        # 生成随机序列
        seq = [random.randint(0, vocab_size - 1) for _ in range(seq_len + 1)]
        
        # 输入是前 seq_len 个 token
        input_seq = seq[:seq_len]
        # 目标是后 seq_len 个 token（向右偏移1）
        target_seq = seq[1:seq_len + 1]
        
        input_seqs.append([float(x) for x in input_seq])
        target_seqs.append([float(x) for x in target_seq])
    
    return input_seqs, target_seqs


def main():
    """主函数。"""
    print("=" * 60)
    print("tinyTorch 框架 - LSTM 序列建模示例")
    print("=" * 60)
    
    # 设置参数
    vocab_size = 50
    embed_dim = 32
    hidden_size = 64
    seq_len = 10
    batch_size = 4
    num_epochs = 3
    learning_rate = 0.001
    
    print("\n步骤 1: 生成模拟序列数据")
    train_inputs, train_targets = generate_toy_sequences(
        vocab_size=vocab_size, 
        seq_len=seq_len, 
        num_samples=80
    )
    val_inputs, val_targets = generate_toy_sequences(
        vocab_size=vocab_size, 
        seq_len=seq_len, 
        num_samples=20
    )
    print(f"  训练样本: {len(train_inputs)}")
    print(f"  验证样本: {len(val_inputs)}")
    print(f"  序列长度: {seq_len}")
    print(f"  词汇表大小: {vocab_size}")
    
    print("\n步骤 2: 创建数据集")
    train_dataset = DataSet(train_inputs, train_targets, batch_size=batch_size, shuffle=True)
    val_dataset = DataSet(val_inputs, val_targets, batch_size=batch_size, shuffle=False)
    print(f"  训练数据集: {train_dataset}")
    print(f"  验证数据集: {val_dataset}")
    
    print("\n步骤 3: 构建 LSTM 语言模型")
    lstm_model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size
    )
    model = Model(name='LSTMLanguageModel', module=lstm_model)
    print(f"  模型: {model.name}")
    print(f"  参数数量: {len(model.parameters())}")
    print(f"  词嵌入维度: {embed_dim}")
    print(f"  LSTM 隐藏层: {hidden_size}")
    
    print("\n步骤 4: 创建优化器和损失函数")
    optimizer = Adam(model.parameters(), learning_rate=learning_rate)
    loss_fn = CrossEntropyLoss()
    print(f"  优化器: {optimizer}")
    print(f"  损失函数: {loss_fn}")
    
    print("\n步骤 5: 训练模型")
    print("注意: 这是演示版本，展示训练流程")
    print("-" * 60)
    
    monitor = Monitor()
    monitor.start()
    
    model.train()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (batch_inputs, batch_targets) in enumerate(train_dataset):
            actual_batch_size = len(batch_inputs)
            
            # 准备输入数据 (batch_size, seq_len)
            input_data = []
            for seq in batch_inputs:
                input_data.extend(seq)
            
            input_tensor = Tensor(input_data, (actual_batch_size, seq_len), 'float32')
            input_var = Variable(input_tensor, requires_grad=True)
            
            # 准备目标数据
            target_data = []
            for seq in batch_targets:
                target_data.extend(seq)
            
            target_tensor = Tensor(target_data, (actual_batch_size * seq_len,), 'float32')
            target_var = Variable(target_tensor, requires_grad=False)
            
            # 前向传播
            optimizer.zero_grad()
            output = lstm_model(input_var)  # (batch_size, seq_len, vocab_size)
            
            # 重塑输出以计算损失
            output_flat_data = output.value.data
            output_flat = Variable(
                Tensor(output_flat_data, (actual_batch_size * seq_len, vocab_size), 'float32'),
                requires_grad=output.requires_grad
            )
            
            # 计算损失
            loss = loss_fn(output_flat, target_var)
            batch_loss = loss.value.data[0]
            epoch_loss += batch_loss
            num_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(train_dataset)}] Loss: {batch_loss:.6f}")
            
            # 省略反向传播（需要完整的自动微分）
            # loss.backward()
            # optimizer.step()
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.6f}")
        
        monitor.record_epoch(epoch + 1, {'train_loss': avg_loss})
    
    print("\n" + "=" * 60)
    print("训练完成!")
    monitor.print_summary()
    
    print("\n步骤 6: 序列生成演示")
    print("注意: 由于未真实训练，生成结果为随机")
    model.eval()
    
    # 生成序列
    seed_seq = [random.randint(0, vocab_size - 1) for _ in range(seq_len)]
    print(f"  种子序列: {seed_seq[:5]}...")
    
    # 创建输入
    input_tensor = Tensor([float(x) for x in seed_seq], (1, seq_len), 'float32')
    input_var = Variable(input_tensor, requires_grad=False)
    
    # 前向传播
    output = lstm_model(input_var)
    
    # 提取最后一个时间步的预测
    last_step_logits = []
    for c in range(vocab_size):
        idx = (seq_len - 1) * vocab_size + c
        last_step_logits.append(output.value.data[idx])
    
    # 选择概率最高的词
    predicted_token = last_step_logits.index(max(last_step_logits))
    print(f"  下一个预测 token: {predicted_token}")
    
    print("\n" + "=" * 60)
    print("LSTM 序列建模示例完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
