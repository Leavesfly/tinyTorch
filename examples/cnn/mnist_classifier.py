"""使用 CNN 进行图像分类的示例。

演示如何使用 tinyTorch 构建简单的卷积神经网络进行图像分类。

Author: TinyAI Team
"""

from tinytorch.tensor import Tensor
from tinytorch.autograd import Variable
from tinytorch.nn import Sequential, Module
from tinytorch.nn.layers import Conv2d, Linear, ReLU, Dropout
from tinytorch.ml import Model, Trainer, DataSet, Adam, CrossEntropyLoss
from tinytorch.ml import AccuracyEvaluator, Monitor
import random


class SimpleCNN(Module):
    """简单的 CNN 分类器。
    
    网络结构：
        Conv2d(1, 16, 3) -> ReLU -> 
        Conv2d(16, 32, 3) -> ReLU ->
        Flatten -> Linear(32*24*24, 128) -> ReLU ->
        Dropout(0.5) -> Linear(128, 10)
    """
    
    def __init__(self, num_classes=10):
        """初始化 CNN。
        
        Args:
            num_classes: 分类数量
        """
        super().__init__()
        
        # 卷积层
        self.conv1 = Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        
        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        
        # 全连接层（假设输入是 28x28）
        self.flatten_size = 32 * 28 * 28
        self.fc1 = Linear(self.flatten_size, 128)
        self.relu3 = ReLU()
        
        self.dropout = Dropout(p=0.5)
        self.fc2 = Linear(128, num_classes)
    
    def forward(self, x: Variable) -> Variable:
        """前向传播。
        
        Args:
            x: 输入图像，形状 (batch_size, 1, 28, 28)
        
        Returns:
            分类 logits，形状 (batch_size, num_classes)
        """
        # 卷积层
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        # 展平
        batch_size = x.value.shape.dims[0]
        flat_data = x.value.data
        flat_tensor = Tensor(flat_data, (batch_size, self.flatten_size), 'float32')
        x = Variable(flat_tensor, requires_grad=x.requires_grad)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def generate_fake_images(num_samples=100, image_size=28, num_classes=10):
    """生成模拟图像数据。
    
    Args:
        num_samples: 样本数量
        image_size: 图像大小（正方形）
        num_classes: 类别数量
    
    Returns:
        (images, labels) 元组
    """
    images = []
    labels = []
    
    for _ in range(num_samples):
        # 生成随机图像数据
        image = [random.uniform(0, 1) for _ in range(image_size * image_size)]
        images.append(image)
        
        # 随机标签
        label = random.randint(0, num_classes - 1)
        labels.append(float(label))
    
    return images, labels


def main():
    """主函数。"""
    print("=" * 60)
    print("tinyTorch 框架 - CNN 图像分类示例")
    print("=" * 60)
    
    # 设置参数
    batch_size = 8
    num_epochs = 3
    learning_rate = 0.001
    num_classes = 10
    image_size = 28
    
    print("\n步骤 1: 生成模拟数据")
    train_images, train_labels = generate_fake_images(num_samples=80, image_size=image_size)
    val_images, val_labels = generate_fake_images(num_samples=20, image_size=image_size)
    print(f"  训练样本: {len(train_images)}")
    print(f"  验证样本: {len(val_images)}")
    
    # 将图像数据重塑为 (batch_size, 1, 28, 28) 格式
    print("\n步骤 2: 准备数据")
    
    # 这里简化处理，实际使用时需要在 DataSet 中处理
    train_data = []
    for img in train_images:
        # 添加通道维度
        train_data.append(img)
    
    val_data = []
    for img in val_images:
        val_data.append(img)
    
    # 创建数据集
    train_dataset = DataSet(train_data, train_labels, batch_size=batch_size, shuffle=True)
    val_dataset = DataSet(val_data, val_labels, batch_size=batch_size, shuffle=False)
    print(f"  训练数据集: {train_dataset}")
    print(f"  验证数据集: {val_dataset}")
    
    print("\n步骤 3: 构建 CNN 模型")
    cnn = SimpleCNN(num_classes=num_classes)
    model = Model(name='SimpleCNN', module=cnn)
    print(f"  模型: {model.name}")
    print(f"  参数数量: {len(model.parameters())}")
    
    print("\n步骤 4: 创建优化器和损失函数")
    optimizer = Adam(model.parameters(), learning_rate=learning_rate)
    loss_fn = CrossEntropyLoss()
    evaluator = AccuracyEvaluator()
    print(f"  优化器: {optimizer}")
    print(f"  损失函数: {loss_fn}")
    print(f"  评估器: {evaluator}")
    
    print("\n步骤 5: 训练模型")
    print("注意: 由于是纯 Python 实现，训练速度较慢，这里仅演示流程")
    print("-" * 60)
    
    monitor = Monitor()
    monitor.start()
    
    # 简化的训练循环（不使用 Trainer，直接演示）
    model.train()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        num_batches = 0
        
        # 训练
        for batch_idx, (batch_data, batch_labels) in enumerate(train_dataset):
            # 将数据转换为 (batch_size, 1, 28, 28) 格式
            actual_batch_size = len(batch_data)
            reshaped_data = []
            for i in range(actual_batch_size):
                reshaped_data.extend(batch_data[i])
            
            # 创建输入张量
            input_tensor = Tensor(reshaped_data, (actual_batch_size, 1, image_size, image_size), 'float32')
            input_var = Variable(input_tensor, requires_grad=True)
            
            # 创建标签张量
            label_tensor = Tensor(batch_labels, (actual_batch_size,), 'float32')
            label_var = Variable(label_tensor, requires_grad=False)
            
            # 前向传播
            optimizer.zero_grad()
            output = cnn(input_var)
            
            # 计算损失
            loss = loss_fn(output, label_var)
            batch_loss = loss.value.data[0]
            epoch_loss += batch_loss
            num_batches += 1
            
            if batch_idx % 2 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(train_dataset)}] Loss: {batch_loss:.6f}")
            
            # 这里省略反向传播和参数更新（因为需要完整的自动微分实现）
            # loss.backward()
            # optimizer.step()
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.6f}")
        
        # 记录指标
        monitor.record_epoch(epoch + 1, {'train_loss': avg_loss})
    
    print("\n" + "=" * 60)
    print("训练完成!")
    monitor.print_summary()
    
    print("\n步骤 6: 模型评估")
    print("注意: 由于未执行真实训练，准确率为随机水平")
    model.eval()
    
    # 简单评估
    predictions = []
    targets = []
    
    for batch_data, batch_labels in val_dataset:
        actual_batch_size = len(batch_data)
        reshaped_data = []
        for i in range(actual_batch_size):
            reshaped_data.extend(batch_data[i])
        
        input_tensor = Tensor(reshaped_data, (actual_batch_size, 1, image_size, image_size), 'float32')
        input_var = Variable(input_tensor, requires_grad=False)
        
        # 前向传播
        output = cnn(input_var)
        
        # 提取预测
        for i in range(actual_batch_size):
            logits = []
            for c in range(num_classes):
                idx = i * num_classes + c
                logits.append(output.value.data[idx])
            predictions.append(logits)
            targets.append(int(batch_labels[i]))
    
    # 计算准确率
    accuracy = evaluator.evaluate(predictions, targets)
    print(f"验证准确率: {accuracy:.2%}")
    
    print("\n" + "=" * 60)
    print("CNN 图像分类示例完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
