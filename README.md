# tinyTorch

<div align="center">

**纯 Python 实现的轻量级深度学习框架**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code](https://img.shields.io/badge/Code-8000%2B%20lines-orange.svg)]()

*一个从零开始实现的深度学习框架，专为教学和学习设计*

[快速开始](#快速开始) • [教程](tutorials/README.md) • [文档](docs/方案.md) • [示例](examples/)

</div>

---

## 📖 项目简介

tinyTorch 是一个完全使用 Python 标准库实现的深度学习框架，不依赖任何第三方库（如 NumPy、PyTorch）。它参考了 [tinyai-deeplearning](https://github.com/Leavesfly/TinyAI/tree/main/tinyai-deeplearning) 项目的架构设计，提供了从张量运算到模型训练的完整功能链路。

### 为什么选择 tinyTorch？

- 🎓 **教学友好**：代码清晰易懂，每一行都能看懂
- 🔍 **深入理解**：揭示深度学习框架的底层实现原理
- 🧩 **完整实现**：包含 Tensor、自动微分、神经网络、训练框架
- 📝 **中文文档**：完整的中文教程和 API 文档
- 🚀 **开箱即用**：提供 CNN、RNN、Transformer 完整示例

## ✨ 核心特性

### 🔥 纯 Python 实现
无任何第三方依赖，所有代码使用 Python 标准库实现，让你清楚了解每个操作的底层逻辑。

### 🏗️ 分层架构设计
采用清晰的四层架构，严格遵循模块化和层次化设计原则：

```
┌─────────────────────────────────────┐
│         应用层 (Application)         │
│      深度学习模型与应用开发           │
├─────────────────────────────────────┤
│    训练层 (tinytorch.ml)             │
│  Model | Trainer | Optimizer | Loss  │
├─────────────────────────────────────┤
│   神经网络层 (tinytorch.nn)          │
│ Module | Linear | Conv | RNN | LSTM  │
├─────────────────────────────────────┤
│   自动微分层 (tinytorch.autograd)    │
│  Variable | Function | Operations    │
├─────────────────────────────────────┤
│    张量层 (tinytorch.tensor)         │
│   Tensor | Shape | Broadcasting      │
└─────────────────────────────────────┘
```

### 🧠 完整的自动微分系统
- 动态计算图构建
- 自动反向传播
- 支持链式求导
- 梯度累积和清零

### 🎯 丰富的神经网络层
- **基础层**：Linear（全连接）
- **卷积层**：Conv2d（二维卷积）
- **循环层**：RNN、LSTM、GRU
- **注意力**：MultiHeadAttention
- **归一化**：LayerNorm、Embedding
- **激活函数**：ReLU、Sigmoid、Tanh

### 📊 完善的训练框架
- Model 模型管理
- Trainer 训练控制
- DataSet 数据处理
- Optimizer（SGD、Adam）
- Loss（MSE、CrossEntropy、BCE）
- Evaluator 模型评估

## 🚀 快速开始

### 安装

```bash
cd tinyTorch
pip install -e .
```

### 5 分钟上手

#### 1. 张量操作

```python
from tinytorch import Tensor

# 创建张量
a = Tensor([[1, 2], [3, 4]])
b = Tensor.zeros((2, 2))
c = Tensor.randn((2, 2))

# 基本运算
result = a + b  # 加法
result = a * c  # 逐元素乘法
result = a.matmul(c)  # 矩阵乘法
```

#### 2. 自动微分

```python
from tinytorch import Variable, Tensor

# 创建变量
x = Variable(Tensor([2.0]), name="x")

# 前向计算
y = x * x + 2 * x + 1  # y = x² + 2x + 1

# 反向传播
y.backward()

# 查看梯度 dy/dx = 2x + 2
print(x.grad)  # [6.0]（当 x=2 时）
```

#### 3. 构建神经网络

```python
from tinytorch.nn import Module, Linear

class SimpleNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(10, 5)
        self.fc2 = Linear(5, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleNet()
```

#### 4. 训练模型

```python
from tinytorch.ml import Model, Trainer, DataSet
from tinytorch.ml.optimizers import SGD
from tinytorch.ml.losses import MSELoss

# 准备数据
train_data = DataSet(X_train, y_train)

# 配置训练
model = Model("simple_net", SimpleNet())
optimizer = SGD(model.parameters(), lr=0.01)
loss_fn = MSELoss()

# 开始训练
trainer = Trainer(
    model=model,
    dataset=train_data,
    optimizer=optimizer,
    loss=loss_fn,
    max_epochs=100,
    batch_size=32
)
trainer.train()
```

完整示例请查看 [快速入门教程](tutorials/01_quickstart.md)

## 📚 模块介绍

### tinytorch.tensor - 张量运算
多维数组的核心实现，支持各种数学运算：

| 组件 | 功能 |
|------|------|
| `Tensor` | 多维数组类，支持创建、索引、变形、运算 |
| `Shape` | 形状管理，处理维度信息和广播 |
| **创建方法** | zeros, ones, randn, rand, eye, arange |
| **基础运算** | add, sub, mul, div, matmul, transpose |
| **数学函数** | exp, log, sqrt, pow, abs |
| **激活函数** | relu, sigmoid, tanh |
| **归约运算** | sum, mean, max, min |

### tinytorch.autograd - 自动微分
动态计算图和自动梯度计算：

| 组件 | 功能 |
|------|------|
| `Variable` | 可微分变量，包含值和梯度 |
| `Function` | 操作基类，定义前向和反向传播 |
| **算术运算** | Add, Sub, Mul, Div, Pow |
| **数学函数** | Exp, Log, Sqrt |
| **激活函数** | ReLU, Sigmoid, Tanh |
| **矩阵运算** | MatMul, Transpose, Reshape |
| **归约运算** | Sum, Mean, Max, Min |

### tinytorch.nn - 神经网络
构建深度学习模型的基础模块：

| 组件 | 功能 |
|------|------|
| `Module` | 所有网络层的基类 |
| `Parameter` | 可训练参数的封装 |
| `Linear` | 全连接层 |
| `Conv2d` | 二维卷积层 |
| `RNN` | 循环神经网络 |
| `LSTM` | 长短期记忆网络 |
| `GRU` | 门控循环单元 |
| `MultiHeadAttention` | 多头注意力机制 |
| `LayerNorm` | 层归一化 |
| `Embedding` | 词嵌入层 |
| `Sequential` | 顺序容器 |

### tinytorch.ml - 机器学习
完整的训练和评估框架：

| 组件 | 功能 |
|------|------|
| `Model` | 模型生命周期管理，支持保存/加载 |
| `Trainer` | 训练循环控制器 |
| `DataSet` | 数据集抽象，支持批处理 |
| `Monitor` | 训练过程监控 |
| **优化器** | SGD, Adam |
| **损失函数** | MSELoss, CrossEntropyLoss, BCELoss |
| **评估器** | AccuracyEvaluator |

## 📖 完整教程

我们提供了一套完整的中文教程，从零开始学习 tinyTorch：

### 入门篇

| 教程 | 内容 | 时长 |
|------|------|------|
| [教程总览](tutorials/README.md) | 学习路径规划和教程导航 | 10min |
| [快速入门](tutorials/01_quickstart.md) | 核心概念和完整示例 | 30min |
| [Tensor 操作详解](tutorials/02_tensor_operations.md) | 张量创建、运算、广播机制 | 1h |
| [自动微分系统](tutorials/03_autograd_system.md) | 计算图、反向传播、梯度检查 | 1.5h |

### 📝 更多教程规划中...
- 构建神经网络
- 训练深度学习模型
- CNN 卷积神经网络
- RNN 循环神经网络
- Transformer 架构

## 💻 完整示例

查看 `examples/` 目录获取完整的可运行示例：

### 已实现的示例

| 示例 | 描述 | 代码行数 |
|------|------|---------|
| [CNN 图像分类](examples/cnn/mnist_classifier.py) | 使用 Conv2d 实现图像分类器 | 257 |
| [LSTM 语言模型](examples/nlp/language_model.py) | 使用 LSTM 进行序列建模 | 270 |
| [Transformer 注意力](examples/transformer/simple_transformer.py) | MultiHeadAttention 实现 | 299 |
| [DeepSeek V3 MoE](examples/deepseek/deepseek_v3_demo.py) | MoE 混合专家架构演示 | 543 |
| [基础测试](examples/basic/quick_test.py) | 快速功能验证 | - |

每个示例都包含：
- ✅ 完整的代码实现
- ✅ 详细的中文注释
- ✅ 模型定义和训练流程
- ✅ 可直接运行

## 📊 项目统计

```
├── 代码文件：42 个 Python 文件
├── 代码行数：约 8,500+ 行
├── 测试文件：3 个（147 行）
├── 示例代码：5 个（1,369 行）
└── 教程文档：4 篇（1,152 行）
```

### 完成度

- ✅ **tensor 模块**：100% 完成
- ✅ **autograd 模块**：100% 完成
- ✅ **nn 模块**：95% 完成（核心层已实现）
- ✅ **ml 模块**：90% 完成（核心功能已实现）
- 📝 **文档**：核心教程已完成，持续更新中

## 🏗️ 项目结构

```
tinyTorch/
├── tinytorch/          # 核心代码包
│   ├── tensor/         # 张量运算（2 个文件）
│   ├── autograd/       # 自动微分（7 个文件）
│   ├── nn/             # 神经网络（10 个文件）
│   ├── ml/             # 机器学习（16 个文件）
│   └── utils/          # 工具函数
├── tests/              # 单元测试
├── examples/           # 完整示例
├── tutorials/          # 学习教程
├── docs/               # 技术文档
└── README.md           # 本文档
```

详细目录树请查看 [技术方案文档](docs/方案.md#41-项目目录树)

## 🆚 与 tinyai-deeplearning 对比

tinyTorch 是 Java 版 tinyai-deeplearning 的 Python 实现：

| 特性 | tinyai-deeplearning (Java) | tinyTorch (Python) |
|------|---------------------------|-------------------|
| **语言** | Java | Python |
| **依赖** | 纯 Java | 纯 Python（仅标准库）|
| **架构** | 4 层（ndarr/func/nnet/ml）| 4 层（tensor/autograd/nn/ml）|
| **设计模式** | 组合、策略、观察者 | 相同 |
| **API 风格** | Java 风格 | Pythonic |
| **性能** | 较快 | 较慢（纯 Python）|
| **定位** | 生产 + 教学 | 教学为主 |
| **文档** | 英文为主 | 中文教程 |

**共同点**：
- 都采用分层架构设计
- 都实现了完整的深度学习功能链路
- 都强调代码可读性和教学价值

## 📚 文档资源

### 技术文档
- [架构设计](docs/architecture.md) - 系统架构和设计理念
- [技术方案](docs/方案.md) - 详细的实现方案（1,425 行）
- [API 参考](docs/api_reference.md) - 完整的 API 文档

### 学习资源
- [教程总览](tutorials/README.md) - 学习路径规划
- [快速入门](tutorials/01_quickstart.md) - 30 分钟上手
- [Tensor 详解](tutorials/02_tensor_operations.md) - 张量操作
- [Autograd 详解](tutorials/03_autograd_system.md) - 自动微分

## 🎯 使用场景

### ✅ 适合用于

- 🎓 学习深度学习框架的底层原理
- 📖 理解自动微分和反向传播机制
- 🔍 研究神经网络层的具体实现
- 👨‍🏫 教学演示和课程项目
- 🧪 算法原型验证

### ❌ 不适合用于

- 🚫 生产环境部署
- 🚫 大规模模型训练
- 🚫 性能要求高的场景
- 🚫 GPU 加速计算

**建议**：学习完 tinyTorch 后，在实际项目中使用 PyTorch 或 TensorFlow。

## 🔧 系统要求

- **Python**：3.7 或更高版本
- **依赖**：无（仅使用 Python 标准库）
- **开发工具**（可选）：
  - pytest（用于运行测试）
  - flake8（用于代码检查）

## 🚦 开始使用

### 1. 克隆项目

```bash
git clone <repository-url>
cd TinyAI/tinyTorch
```

### 2. 安装

```bash
pip install -e .
```

### 3. 运行示例

```bash
# 快速测试
python examples/basic/quick_test.py

# CNN 示例
python examples/cnn/mnist_classifier.py

# LSTM 示例
python examples/nlp/language_model.py

# Transformer 示例
python examples/transformer/simple_transformer.py
```

### 4. 学习教程

从 [快速入门教程](tutorials/01_quickstart.md) 开始你的学习之旅！

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

### 贡献方式
- 🐛 提交 Issue 报告问题
- 💡 提出新功能建议
- 📝 改进文档和教程
- 🔧 提交 Pull Request

### 开发流程
1. Fork 项目
2. 创建特性分支
3. 提交你的修改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- 灵感来源：[tinyai-deeplearning](../tinyai-deeplearning) 项目
- 设计参考：PyTorch 框架
- 教学目标：与 TinyAI 项目保持一致

## 📮 联系方式

- 项目主页：[TinyAI](https://github.com/leavesfly/TinyAI)
- 问题反馈：请在项目中提交 Issue
- 讨论交流：欢迎参与项目讨论

---

<div align="center">

**⭐ 如果这个项目对你有帮助，欢迎 Star！**

[返回顶部](#tinytorch) • [教程](tutorials/README.md) • [文档](docs/方案.md) • [示例](examples/)

Made with ❤️ by TinyAI Team

</div>
