"""简单的线性回归示例。

演示如何使用 tinyTorch 框架训练一个简单的线性回归模型。

Author: TinyAI Team
"""

import random


def generate_linear_data(num_samples=100, input_dim=1):
    """生成线性回归数据。
    
    y = 2*x + 1 + noise
    
    Args:
        num_samples: 样本数量
        input_dim: 输入维度
    
    Returns:
        (data, labels) 元组
    """
    data = []
    labels = []
    
    for _ in range(num_samples):
        # 生成随机输入
        x = [random.uniform(-10, 10) for _ in range(input_dim)]
        
        # 计算输出: y = 2*x + 1 + noise
        y = 2 * x[0] + 1 + random.gauss(0, 0.5)
        
        data.append(x)
        labels.append([y])
    
    return data, labels


def main():
    """主函数。"""
    print("=" * 60)
    print("tinyTorch 框架 - 线性回归示例")
    print("=" * 60)
    
    # 导入必要的模块
    from tinytorch.nn import Sequential, Linear
    from tinytorch.ml import Model, Trainer, DataSet, SGD, MSELoss
    
    # 1. 生成训练数据
    print("\n步骤 1: 生成训练数据")
    train_data, train_labels = generate_linear_data(num_samples=200, input_dim=1)
    val_data, val_labels = generate_linear_data(num_samples=50, input_dim=1)
    print(f"  训练样本数: {len(train_data)}")
    print(f"  验证样本数: {len(val_data)}")
    
    # 2. 创建数据集
    print("\n步骤 2: 创建数据集")
    train_dataset = DataSet(train_data, train_labels, batch_size=32, shuffle=True)
    val_dataset = DataSet(val_data, val_labels, batch_size=32, shuffle=False)
    print(f"  训练数据集: {train_dataset}")
    print(f"  验证数据集: {val_dataset}")
    
    # 3. 构建模型
    print("\n步骤 3: 构建模型")
    net = Sequential(
        Linear(1, 1)  # 简单的线性层: y = w*x + b
    )
    model = Model(name='LinearRegression', module=net)
    print(f"  模型: {model.name}")
    print(f"  网络结构:\n{net}")
    
    # 4. 创建优化器和损失函数
    print("\n步骤 4: 创建优化器和损失函数")
    optimizer = SGD(model.parameters(), learning_rate=0.01)
    loss_fn = MSELoss()
    print(f"  优化器: {optimizer}")
    print(f"  损失函数: {loss_fn}")
    
    # 5. 创建训练器
    print("\n步骤 5: 创建训练器")
    trainer = Trainer(
        model=model,
        dataset=train_dataset,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=10,
        print_interval=2,
        val_dataset=val_dataset
    )
    print(f"  训练器: {trainer}")
    
    # 6. 开始训练
    print("\n步骤 6: 开始训练")
    trainer.train()
    
    # 7. 查看训练结果
    print("\n步骤 7: 查看训练结果")
    print(f"  最终训练损失: {trainer.train_losses[-1]:.6f}")
    print(f"  最终验证损失: {trainer.val_losses[-1]:.6f}")
    
    # 8. 查看学到的参数
    print("\n步骤 8: 查看学到的参数（理论值: w=2, b=1）")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.value.data[:5]}...")  # 只打印前几个值
    
    # 9. 保存模型
    print("\n步骤 9: 保存模型")
    model.save_parameters('linear_regression_model.pkl')
    print("  模型已保存至: linear_regression_model.pkl")
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
