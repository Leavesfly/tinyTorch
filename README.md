# tinyTorch： A Tiny PyTorch

纯 Python、仅依赖标准库的轻量级深度学习框架，适合用来学习 `NdArray`、自动微分、神经网络层以及完整训练流程是如何从零搭起来的。

项目当前已经覆盖从张量运算到训练工具链的主要能力，并补齐了卷积、序列模型、注意力、交叉熵损失、PyTorch 风格 `DataLoader`、统一随机数工具等模块。但它仍然是一个以教学和原理验证为主的 CPU-only 项目，不追求与 NumPy / PyTorch 同级的性能。

## 项目定位

`tinyTorch` 适合你在这些场景里使用：

- 学习深度学习框架的底层实现
- 阅读自动微分和反向传播的具体代码
- 验证小规模模型或算法原型
- 教学演示、课程作业、博客示例

它不适合：

- 大规模训练
- 生产环境部署
- GPU 加速场景
- 对吞吐和数值性能要求很高的任务

## 当前能力

- `tinytorch.ndarr`
  - `NdArray`、`Shape`
  - 嵌套列表建模、广播、reshape、transpose、matmul、reduce、常见数学函数
  - `zeros` / `ones` / `randn` / `uniform` 等创建接口
- `tinytorch.autograd`
  - `Tensor`、`Function`、`no_grad()`
  - 动态计算图
  - 自动反向传播与梯度累积
- `tinytorch.nn`
  - 基础模块：`Module`、`Parameter`、`Sequential`、`ModuleList`
  - 常见层：`Linear`、`ReLU`、`Sigmoid`、`Tanh`、`LeakyReLU`
  - 归一化与正则：`LayerNorm`、`Dropout`
  - 高级层：`Embedding`、`Conv2d`、`RNN`、`LSTM`、`GRU`、`MultiHeadAttention`
- `tinytorch.ml`
  - `Model`、`Trainer`、`DataSet`
  - 优化器：`SGD`、`Adam`
  - 损失函数：`MSELoss`、`CrossEntropyLoss`、`BCELoss`
  - 训练辅助：`Monitor`、`EarlyStopping`、评估器
- `tinytorch.utils`
  - `Dataset` / `IterableDataset` / `DataLoader`
  - `Sampler` / `BatchSampler`
  - 统一随机数入口：`tinytorch.utils.random`

## 分层结构

```text
tinyTorch
├── tinytorch.ndarr      # 多维数组与广播、形状、基础数值计算
├── tinytorch.autograd   # Tensor、Function、动态计算图、反向传播
├── tinytorch.nn         # Module/Parameter 以及常见神经网络层
├── tinytorch.ml         # 训练循环、优化器、损失函数、模型保存加载
└── tinytorch.utils      # DataLoader、Sampler、随机数工具
```

## 安装

```bash
git clone <repository-url>
cd tinyTorch
pip install -e .
```

Python 要求为 `3.7+`。

如果你想运行测试：

```bash
pip install -e ".[dev]"
pytest
```

## 快速开始

### 1. `NdArray` 与自动微分

```python
from tinytorch import NdArray, Tensor

x = Tensor(NdArray([2.0]), name="x")
y = x * x + 2 * x + 1

y.backward()

print("y =", y.value.data)   # [9.0]
print("dy/dx =", x.grad.data)  # [6.0]
```

### 2. 训练一个简单 MLP

```python
from tinytorch.nn import Sequential, Linear, ReLU
from tinytorch.ml import Model, Trainer, DataSet
from tinytorch.ml.optimizers import SGD
from tinytorch.ml.losses import MSELoss

net = Sequential(
    Linear(2, 4),
    ReLU(),
    Linear(4, 1),
)

model = Model("xor_mlp", net)
dataset = DataSet(
    data=[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    labels=[[0.0], [1.0], [1.0], [0.0]],
    batch_size=2,
    shuffle=True,
)
optimizer = SGD(model.parameters(), lr=0.1)
loss_fn = MSELoss()

trainer = Trainer(
    model=model,
    dataset=dataset,
    optimizer=optimizer,
    loss_fn=loss_fn,
    max_epochs=20,
    print_interval=1,
)
trainer.train()
```

### 3. 使用嵌入、注意力和序列模型

高级层当前从 `tinytorch.nn.layers` 导出：

```python
from tinytorch import NdArray
from tinytorch.autograd import Tensor
from tinytorch.nn.layers import Embedding, MultiHeadAttention, LSTM

tokens = Tensor(NdArray([[1, 2, 3], [3, 2, 1]]), requires_grad=False)

embedding = Embedding(num_embeddings=10, embedding_dim=8)
attention = MultiHeadAttention(embed_dim=8, num_heads=2, dropout=0.0)
lstm = LSTM(input_size=8, hidden_size=8)

x = embedding(tokens)    # (batch, seq, embed_dim)
x = attention(x)         # self-attention
y, c = lstm(x)           # y: all hidden states, c: final cell state

print(x.value.shape.dims)
print(y.value.shape.dims)
print(c.value.shape.dims)
```

### 4. 固定随机种子

```python
from tinytorch.utils import random as tt_random
from tinytorch.ndarr import NdArray

tt_random.seed(123)
a = NdArray.randn((2, 2))

tt_random.seed(123)
b = NdArray.randn((2, 2))

assert a.data == b.data
```

## 导入建议

为了避免 README 和实际导出接口再次脱节，推荐按下面的方式使用：

- 核心数据结构从顶层导入：`from tinytorch import NdArray, Tensor, Function, no_grad`
- 常见基础层从 `tinytorch.nn` 导入：`Linear`、`ReLU`、`Sequential` 等
- 高级层从 `tinytorch.nn.layers` 导入：`Conv2d`、`RNN`、`LSTM`、`GRU`、`MultiHeadAttention`
- 训练相关从 `tinytorch.ml` 或其子模块导入
- 数据加载与随机工具从 `tinytorch.utils` 导入

## 模块概览

### `tinytorch.ndarr`

数组核心层，负责数据存储、形状管理和广播语义。`NdArray` 使用扁平列表存储底层数据，适合作为理解张量库实现细节的切入点。

### `tinytorch.autograd`

在 `NdArray` 之上构建动态计算图。`Tensor` 负责追踪依赖关系，`Function` 负责定义前向和反向规则。标量输出可以直接 `backward()`，非标量输出需要显式提供 `grad_output`。

### `tinytorch.nn`

提供 `Module` / `Parameter` 抽象，以及线性层、卷积层、归一化、嵌入、RNN/LSTM/GRU、多头注意力等常见神经网络组件。

### `tinytorch.ml`

封装训练流程和模型管理能力，包括：

- `Model.state_dict()` / `load_state_dict()`
- `Model.save()` / `Model.load()`
- `Trainer` 训练循环
- `SGD` / `Adam`
- `MSELoss` / `CrossEntropyLoss` / `BCELoss`

### `tinytorch.utils`

提供两类实用工具：

- 旧的训练辅助数据接口：`tinytorch.ml.DataSet`
- 更接近 PyTorch 的 `Dataset` / `DataLoader` / `Sampler`

## 文档与示例

- 教程总览：`docs/tutorials/README.md`
- 快速入门：`docs/tutorials/01_quickstart.md`
- `NdArray` 教程：`docs/tutorials/02_tensor_operations.md`
- 自动微分教程：`docs/tutorials/03_autograd_system.md`
- 技术方案：`docs/方案.md`

可运行示例位于 `examples/`：

- `examples/basic/quick_test.py`
- `examples/basic/linear_regression.py`
- `examples/cnn/mnist_classifier.py`
- `examples/nlp/language_model.py`
- `examples/transformer/simple_transformer.py`
- `examples/deepseek/deepseek_v3_demo.py`

## 当前限制与注意事项

- 这是一个教学型、Alpha 阶段项目，很多算子采用纯 Python 循环实现，性能会明显低于 NumPy / PyTorch。
- `Tensor` 只能包装 `NdArray`，不能直接包装原始 Python 列表。
- `Tensor.backward()` 对非标量输出必须手动传入 `grad_output`。
- `CrossEntropyLoss` 期望输入是 `logits`，形状通常为 `(batch_size, num_classes)`；目标是类别索引，形状通常为 `(batch_size,)`。
- `BCELoss` 期望输入是概率值，不是 logits。
- `Model.load()` 目前需要先手动构造模型结构，再通过 `module=...` 加载参数。
- `DataLoader` 目前仅支持单进程；`num_workers > 0` 和 `pin_memory=True` 会抛出 `NotImplementedError`。
- 顶层 `tinytorch.nn` 没有导出全部高级层；如需卷积、循环层、注意力，请从 `tinytorch.nn.layers` 导入。

## 开发

```bash
pytest
```

如果你准备继续扩展框架，建议优先阅读：

1. `tinytorch/ndarr/ndarray.py`
2. `tinytorch/autograd/tensor.py`
3. `tinytorch/nn/module.py`
4. `tests/`

## 许可证

本项目使用 `MIT` 许可证，详见 `LICENSE`。
