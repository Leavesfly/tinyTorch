"""训练可视化示例。

演示如何使用 tinyTorch 的 TrainingVisualizer 在训练过程中：
- 实时查看损失曲线、调整显示参数、回放训练动画
- 查看模型的动态计算图结构（D3.js 力导向图）
- 查看模块层级结构
- 导出图表和计算图

运行方式：
    python examples/visualization/train_visualizer_demo.py

启动后会自动打开浏览器：
- http://localhost:8097       → 训练指标可视化
- http://localhost:8097/graph → 动态计算图可视化

Author: TinyAI Team
"""

import random
import math
import time


def generate_regression_data(num_samples=300):
    """生成非线性回归数据。

    y = sin(x1) + 0.5 * x2 + noise

    Args:
        num_samples: 样本数量

    Returns:
        (data, labels) 元组
    """
    data = []
    labels = []

    for _ in range(num_samples):
        x1 = random.uniform(-3, 3)
        x2 = random.uniform(-2, 2)
        y = math.sin(x1) + 0.5 * x2 + random.gauss(0, 0.2)
        data.append([x1, x2])
        labels.append([y])

    return data, labels


def main():
    """主函数：带可视化的训练流程。"""
    print("=" * 60)
    print("🔥 tinyTorch - 训练 & 计算图可视化示例")
    print("=" * 60)

    from tinytorch import NdArray, Tensor
    from tinytorch.nn import Sequential, Linear, ReLU
    from tinytorch.ml import Model, Trainer, DataSet, SGD, MSELoss
    from tinytorch.ml import TrainingVisualizer

    # ── 1. 准备数据 ──────────────────────────────────────────
    print("\n📦 步骤 1: 生成训练数据")
    train_data, train_labels = generate_regression_data(num_samples=300)
    val_data, val_labels = generate_regression_data(num_samples=80)
    print(f"  训练样本: {len(train_data)}, 验证样本: {len(val_data)}")

    train_dataset = DataSet(train_data, train_labels, batch_size=32, shuffle=True)
    val_dataset = DataSet(val_data, val_labels, batch_size=32, shuffle=False)

    # ── 2. 构建模型 ──────────────────────────────────────────
    print("\n🏗️  步骤 2: 构建模型")
    net = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 8),
        ReLU(),
        Linear(8, 1),
    )
    model = Model(name='VisualDemo_MLP', module=net)
    print(f"  模型: {model.name}")

    optimizer = SGD(model.parameters(), learning_rate=0.01)
    loss_fn = MSELoss()

    # ── 3. 启动可视化器 ──────────────────────────────────────
    print("\n📊 步骤 3: 启动可视化服务")
    viz = TrainingVisualizer(port=8097, auto_open=True)
    viz.start_server()

    # ── 4. 可视化计算图 ──────────────────────────────────────
    #    做一次前向传播来构建计算图，然后提取并可视化
    print("\n🔍 步骤 4: 可视化动态计算图")
    sample_input = Tensor(NdArray([train_data[0]]), requires_grad=False)
    sample_target = Tensor(NdArray([train_labels[0]]), requires_grad=False)
    sample_output = model(sample_input)
    sample_loss = loss_fn(sample_output, sample_target)

    # 将计算图和模块结构注册到可视化器
    # 注意：必须在 backward() 之前调用，否则 creator 链会被清除
    viz.set_graph(output_tensor=sample_loss, module=net)
    print("  计算图已注册，访问 http://localhost:8097/graph 查看")

    # 清理这次前向传播的梯度
    optimizer.zero_grad()

    # ── 5. 开始训练 ──────────────────────────────────────────
    print("\n🚀 步骤 5: 开始训练")
    trainer = Trainer(
        model=model,
        dataset=train_dataset,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=50,
        print_interval=5,
        val_dataset=val_dataset,
        visualizer=viz,
    )
    trainer.train()

    # ── 6. 导出可视化结果 ─────────────────────────────────────
    print("\n💾 步骤 6: 导出可视化结果")
    viz.save_data('training_data.json')
    viz.export_html('training_report.html')

    # ── 7. 保持服务运行 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ 训练完成！可视化服务仍在运行。")
    print("   你可以在浏览器中查看：")
    print("     📊 http://localhost:8097       → 训练指标可视化")
    print("     🔍 http://localhost:8097/graph → 动态计算图")
    print("")
    print("   训练指标页面功能：")
    print("     - 拖动 Smoothing 滑块平滑损失曲线")
    print("     - 切换 Y-Axis Scale 为 Log 查看对数尺度")
    print("     - 点击 Play 按钮回放训练过程动画")
    print("     - 点击 Export PNG/SVG 导出图表")
    print("")
    print("   计算图页面功能：")
    print("     - 拖拽节点调整布局")
    print("     - 滚轮缩放、平移画布")
    print("     - 点击节点查看详细信息（形状、参数等）")
    print("     - 切换 Computation Graph / Module Structure 标签页")
    print("     - 导出为 SVG/PNG")
    print("")
    print("   按 Ctrl+C 退出。")
    print("=" * 60)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 正在关闭可视化服务...")
        viz.stop_server()
        print("已退出。")


if __name__ == '__main__':
    main()
