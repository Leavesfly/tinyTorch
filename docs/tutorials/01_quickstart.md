# tinyTorch 快速入门

欢迎使用 tinyTorch！这是一个纯 Python 实现的轻量级深度学习框架，专为教学和学习设计。

## 什么是 tinyTorch？

tinyTorch 是一个从零开始实现的深度学习框架，不依赖任何第三方库（如 NumPy、PyTorch），完全使用 Python 标准库实现。它的目标是帮助你深入理解深度学习框架的内部工作原理。

## 安装

```bash
cd tinyTorch
pip install -e .
```

## 核心概念

### 1. Tensor（张量）

Tensor 是 tinyTorch 的基础数据结构，类似于多维数组。

```python
from tinytorch import Tensor

# 创建张量
a = Tensor([1, 2, 3, 4])          # 一维张量
b = Tensor([[1, 2], [3, 4]])      # 二维张量

# 工厂方法
zeros = Tensor.zeros((2, 3))      # 全零张量
ones = Tensor.ones((3, 2))        # 全一张量
randn = Tensor.randn((2, 2))      # 随机张量

# 查看形状
print(a.shape.dims)               # (4,)
print(b.shape.dims)               # (2, 2)
```

### 2. Variable（变量）

Variable 是带有梯度信息的张量，用于自动微分。

```python
from tinytorch import Variable, Tensor

# 创建变量
x = Variable(Tensor([1.0, 2.0, 3.0]), name="x")

# 计算
y = x * 2 + 1

# 反向传播
y.backward()

# 查看梯度
print(x.grad)  # dy/dx
```

### 3. Module（模块）

Module 是神经网络层的基类。

```python
from tinytorch.nn import Module, Linear

class SimpleNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(10, 5)
        self.fc2 = Linear(5, 2)
    
    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleNet()
```

### 4. 训练模型

```python
from tinytorch.ml import Model, Trainer, DataSet
from tinytorch.ml.optimizers import SGD
from tinytorch.ml.losses import MSELoss

# 准备数据
train_data = DataSet(X_train, y_train)

# 创建模型
model = Model("simple_net", SimpleNet())

# 配置训练
optimizer = SGD(model.parameters(), lr=0.01)
loss_fn = MSELoss()

# 训练
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

## 完整示例：线性回归

```python
from tinytorch import Tensor, Variable
from tinytorch.nn import Module, Linear
from tinytorch.ml import Model, Trainer, DataSet
from tinytorch.ml.optimizers import SGD
from tinytorch.ml.losses import MSELoss

# 1. 准备数据
X_train = [[1.0], [2.0], [3.0], [4.0]]
y_train = [[2.0], [4.0], [6.0], [8.0]]
train_data = DataSet(X_train, y_train)

# 2. 定义模型
class LinearRegression(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# 3. 创建模型和训练器
model = Model("linear_regression", LinearRegression())
optimizer = SGD(model.parameters(), lr=0.01)
loss_fn = MSELoss()

trainer = Trainer(
    model=model,
    dataset=train_data,
    optimizer=optimizer,
    loss=loss_fn,
    max_epochs=100,
    batch_size=2
)

# 4. 训练
trainer.train()

# 5. 预测
x_test = Variable(Tensor([[5.0]]))
y_pred = model(x_test)
print(f"预测结果: {y_pred.value.data}")
```

## 下一步

- [02_tensor_operations.md](02_tensor_operations.md) - 深入学习张量操作
- [03_autograd_system.md](03_autograd_system.md) - 理解自动微分系统
- [04_building_networks.md](04_building_networks.md) - 构建神经网络
- [05_training_models.md](05_training_models.md) - 训练深度学习模型

## 常见问题

**Q: 为什么不使用 NumPy？**  
A: tinyTorch 的目标是教学。通过纯 Python 实现，你可以看到每一行代码，理解底层原理。

**Q: 性能如何？**  
A: 纯 Python 实现性能较慢，不适合生产环境。它是为学习设计的。

**Q: 可以用于实际项目吗？**  
A: 不建议。对于实际项目，请使用 PyTorch 或 TensorFlow。

## 参考资源

- [示例代码](../examples/) - 更多完整示例
- [API 文档](../docs/api_reference.md) - 详细 API 说明
- [架构设计](../docs/architecture.md) - 框架设计思想
