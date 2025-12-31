# 自动微分系统详解

本教程深入讲解 tinyTorch 的自动微分（Automatic Differentiation）系统。

## 什么是自动微分？

自动微分是深度学习框架的核心功能，它能够自动计算函数的导数。在训练神经网络时，我们需要计算损失函数对参数的梯度，自动微分系统使这一过程变得简单高效。

## Variable 和 Function

### Variable（变量）

Variable 是带有梯度信息的张量包装器。

```python
from tinytorch import Variable, Tensor

# 创建变量
x = Variable(Tensor([1.0, 2.0, 3.0]), name="x")

print(x.value)        # 查看值
print(x.grad)         # 查看梯度（初始为 None）
print(x.requires_grad)  # 是否需要梯度（默认 True）
```

### Function（函数）

Function 是所有可微分操作的基类。每个操作都需要实现：
- `forward()`: 前向计算
- `backward()`: 反向传播梯度

## 计算图

自动微分系统通过构建计算图来跟踪操作。

```python
from tinytorch import Variable, Tensor

# 构建计算图
x = Variable(Tensor([2.0]), name="x")
y = Variable(Tensor([3.0]), name="y")

# z = x^2 + 2*x*y + y^2
z = x * x + x * y * 2 + y * y

# 查看计算图
print(z.creator)      # 创建 z 的函数
```

计算图示例：
```
x ─┬─> Square ─┬─> Add ─┬─> Add ──> z
   │           │        │
   └─> Mul ────┘        │
   └────────────────────┘
y ─┬─> Mul ─────────────┘
   └─> Square ──────────┘
```

## 反向传播

### 基本用法

```python
from tinytorch import Variable, Tensor

# 前向传播
x = Variable(Tensor([3.0]), name="x")
y = x * x + 2 * x + 1

# 反向传播
y.backward()

# 查看梯度 dy/dx = 2x + 2
print(x.grad)  # [8.0] (当 x=3 时)
```

### 链式法则

自动微分系统自动应用链式法则计算梯度。

```python
from tinytorch import Variable, Tensor

# f(x) = (x^2 + 1)^3
x = Variable(Tensor([2.0]), name="x")
y = x * x + 1     # y = x^2 + 1
z = y * y * y     # z = y^3

z.backward()

# df/dx = 3(x^2+1)^2 * 2x
# 当 x=2 时: 3*(5)^2 * 4 = 300
print(x.grad)  # [300.0]
```

## 梯度累积

默认情况下，梯度会累积。

```python
from tinytorch import Variable, Tensor

x = Variable(Tensor([1.0]), name="x")

# 第一次反向传播
y1 = x * 2
y1.backward()
print(x.grad)  # [2.0]

# 第二次反向传播（梯度累积）
y2 = x * 3
y2.backward()
print(x.grad)  # [5.0] = 2 + 3

# 清零梯度
x.zero_grad()
print(x.grad)  # [0.0]
```

## 常用操作的梯度

### 算术运算

```python
from tinytorch import Variable, Tensor

# 加法: d(x+y)/dx = 1, d(x+y)/dy = 1
x = Variable(Tensor([1.0]))
y = Variable(Tensor([2.0]))
z = x + y
z.backward()
print(x.grad, y.grad)  # [1.0], [1.0]

# 乘法: d(xy)/dx = y, d(xy)/dy = x
x = Variable(Tensor([3.0]))
y = Variable(Tensor([4.0]))
z = x * y
z.backward()
print(x.grad, y.grad)  # [4.0], [3.0]
```

### 数学函数

```python
from tinytorch import Variable, Tensor
import math

# 指数: d(e^x)/dx = e^x
x = Variable(Tensor([1.0]))
y = x.exp()
y.backward()
print(x.grad)  # [2.718...] = e^1

# 对数: d(ln(x))/dx = 1/x
x = Variable(Tensor([2.0]))
y = x.log()
y.backward()
print(x.grad)  # [0.5] = 1/2

# 幂函数: d(x^n)/dx = n*x^(n-1)
x = Variable(Tensor([3.0]))
y = x ** 2
y.backward()
print(x.grad)  # [6.0] = 2*3
```

### 激活函数

```python
from tinytorch import Variable, Tensor

# ReLU: d(ReLU(x))/dx = 1 if x>0 else 0
x = Variable(Tensor([2.0, -1.0]))
y = x.relu()
y.backward()
print(x.grad)  # [1.0, 0.0]

# Sigmoid: d(σ(x))/dx = σ(x)*(1-σ(x))
x = Variable(Tensor([0.0]))
y = x.sigmoid()
y.backward()
print(x.grad)  # [0.25] = 0.5*0.5

# Tanh: d(tanh(x))/dx = 1 - tanh(x)^2
x = Variable(Tensor([0.0]))
y = x.tanh()
y.backward()
print(x.grad)  # [1.0] = 1 - 0^2
```

### 矩阵运算

```python
from tinytorch import Variable, Tensor

# 矩阵乘法梯度
A = Variable(Tensor([[1.0, 2.0]]))  # (1, 2)
B = Variable(Tensor([[3.0], [4.0]]))  # (2, 1)
C = A.matmul(B)  # (1, 1)

C.backward()
# dC/dA = B^T
print(A.grad)  # [[3.0, 4.0]]
# dC/dB = A^T
print(B.grad)  # [[1.0], [2.0]]
```

## 梯度检查

验证自动微分的正确性。

```python
from tinytorch import Variable, Tensor

def numerical_gradient(f, x, eps=1e-5):
    """数值梯度（有限差分法）"""
    x_val = x.value.data[0]
    
    # f(x + eps)
    x.value.data[0] = x_val + eps
    f_plus = f(x).value.data[0]
    
    # f(x - eps)
    x.value.data[0] = x_val - eps
    f_minus = f(x).value.data[0]
    
    # 恢复原值
    x.value.data[0] = x_val
    
    # 数值梯度
    return (f_plus - f_minus) / (2 * eps)

# 测试函数 f(x) = x^2
def f(x):
    return x * x

x = Variable(Tensor([3.0]))
y = f(x)
y.backward()

# 比较自动微分和数值梯度
auto_grad = x.grad.data[0]
num_grad = numerical_gradient(f, x)

print(f"自动微分梯度: {auto_grad}")  # 6.0
print(f"数值梯度: {num_grad}")        # ~6.0
print(f"误差: {abs(auto_grad - num_grad)}")  # 很小
```

## Detach（分离）

有时我们需要阻止梯度传播。

```python
from tinytorch import Variable, Tensor

x = Variable(Tensor([2.0]), name="x")
y = x * x  # y = x^2

# 分离 y，阻止梯度传播到 x
y_detached = y.detach()
z = y_detached * 3

z.backward()

# y 不会传播梯度到 x
print(x.grad)  # None 或 [0.0]
```

## 实践示例：手动实现梯度下降

```python
from tinytorch import Variable, Tensor

# 目标：最小化 f(x) = (x - 3)^2

# 初始化参数
x = Variable(Tensor([0.0]), name="x")

# 梯度下降
learning_rate = 0.1
for epoch in range(50):
    # 前向传播
    loss = (x - 3) ** 2
    
    # 反向传播
    x.zero_grad()
    loss.backward()
    
    # 更新参数
    x.value.data[0] -= learning_rate * x.grad.data[0]
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: x={x.value.data[0]:.4f}, "
              f"loss={loss.value.data[0]:.4f}")

# 最终结果应接近 3
print(f"最终 x: {x.value.data[0]:.4f}")
```

## 实践示例：多变量优化

```python
from tinytorch import Variable, Tensor

# 目标：最小化 f(x,y) = (x-2)^2 + (y-3)^2

x = Variable(Tensor([0.0]), name="x")
y = Variable(Tensor([0.0]), name="y")

learning_rate = 0.1
for epoch in range(100):
    # 前向传播
    loss = (x - 2) ** 2 + (y - 3) ** 2
    
    # 反向传播
    x.zero_grad()
    y.zero_grad()
    loss.backward()
    
    # 更新参数
    x.value.data[0] -= learning_rate * x.grad.data[0]
    y.value.data[0] -= learning_rate * y.grad.data[0]
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: x={x.value.data[0]:.4f}, "
              f"y={y.value.data[0]:.4f}, loss={loss.value.data[0]:.4f}")

print(f"最终结果: x={x.value.data[0]:.4f}, y={y.value.data[0]:.4f}")
```

## 高级话题

### 高阶导数

```python
from tinytorch import Variable, Tensor

# 计算二阶导数
x = Variable(Tensor([2.0]), name="x")

# f(x) = x^3
y = x ** 3

# 一阶导数 f'(x) = 3x^2
y.backward()
first_grad = x.grad.data[0]  # 12 (当 x=2)

# 二阶导数需要重新构建计算图
# 这在当前实现中较复杂，通常用数值方法
```

### 自定义操作

```python
from tinytorch.autograd import Function, Variable
from tinytorch import Tensor

class MySquare(Function):
    """自定义平方操作"""
    
    def forward(self, x):
        self.save_for_backward(x)
        result = x.value.mul(x.value)
        return Variable(result, creator=self)
    
    def backward(self, grad_output):
        x, = self.saved_tensors
        # d(x^2)/dx = 2x
        grad_input = x.value.mul_scalar(2).mul(grad_output.value)
        return Variable(grad_input)

# 使用自定义操作
x = Variable(Tensor([3.0]))
square_fn = MySquare()
y = square_fn(x)
y.backward()
print(x.grad)  # [6.0]
```

## 调试技巧

### 打印计算图

```python
from tinytorch import Variable, Tensor

def print_graph(var, indent=0):
    """打印计算图结构"""
    prefix = "  " * indent
    print(f"{prefix}{var.name or 'unnamed'}: {var.value.shape.dims}")
    
    if var.creator:
        print(f"{prefix}  <- {var.creator.__class__.__name__}")
        for input_var in var.creator.inputs:
            print_graph(input_var, indent + 2)

x = Variable(Tensor([1.0]), name="x")
y = Variable(Tensor([2.0]), name="y")
z = (x + y) * (x - y)

print_graph(z)
```

### 梯度检查工具

```python
def check_gradient(f, x, eps=1e-5, tolerance=1e-3):
    """检查梯度正确性"""
    # 自动微分
    y = f(x)
    y.backward()
    auto_grad = x.grad.data[0]
    
    # 数值梯度
    num_grad = numerical_gradient(f, x, eps)
    
    # 比较
    error = abs(auto_grad - num_grad)
    if error < tolerance:
        print(f"✓ 梯度检查通过 (误差: {error:.2e})")
    else:
        print(f"✗ 梯度检查失败 (误差: {error:.2e})")
        print(f"  自动微分: {auto_grad}")
        print(f"  数值梯度: {num_grad}")
```

## 常见问题

**Q: 为什么梯度是累积的？**  
A: 这在某些场景下很有用（如梯度累积训练）。记得在每次优化步骤前调用 `zero_grad()`。

**Q: 如何处理原地操作？**  
A: 原地操作会破坏计算图，应该避免。使用返回新对象的操作。

**Q: Variable 和 Tensor 的区别？**  
A: Tensor 只存储数据，Variable 额外存储梯度和计算图信息。

## 下一步

- [04_building_networks.md](04_building_networks.md) - 构建神经网络
- [05_training_models.md](05_training_models.md) - 训练深度学习模型

## 练习题

1. 实现一个函数，验证 Sigmoid 激活函数的梯度公式
2. 使用自动微分实现线性回归的梯度下降
3. 实现一个自定义的 Softmax 操作，包括前向和反向传播
4. 编写一个梯度检查工具，测试所有内置操作
