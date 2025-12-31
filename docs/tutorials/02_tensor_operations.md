# Tensor 操作详解

本教程深入讲解 tinyTorch 中的张量（Tensor）操作。

## 什么是 Tensor？

Tensor 是多维数组的数学表示，是深度学习的基础数据结构。在 tinyTorch 中，Tensor 使用扁平列表存储数据，通过 Shape 对象管理维度。

## 创建 Tensor

### 从列表创建

```python
from tinytorch import Tensor

# 一维张量
t1 = Tensor([1, 2, 3, 4])
print(t1.shape.dims)  # (4,)

# 二维张量
t2 = Tensor([[1, 2, 3], 
             [4, 5, 6]])
print(t2.shape.dims)  # (2, 3)

# 三维张量
t3 = Tensor([[[1, 2], [3, 4]], 
             [[5, 6], [7, 8]]])
print(t3.shape.dims)  # (2, 2, 2)
```

### 使用工厂方法

```python
from tinytorch import Tensor

# 全零张量
zeros = Tensor.zeros((3, 4))

# 全一张量
ones = Tensor.ones((2, 3))

# 随机张量（正态分布）
randn = Tensor.randn((2, 2))

# 随机张量（均匀分布）
rand = Tensor.rand((3, 3))

# 单位矩阵
eye = Tensor.eye(3)

# 等差数列
arange = Tensor.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
```

## 基本运算

### 算术运算

```python
from tinytorch import Tensor

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

# 加法
c = a.add(b)          # [5, 7, 9]
c = a + b             # 使用运算符重载

# 减法
c = a.sub(b)          # [-3, -3, -3]
c = a - b

# 乘法（逐元素）
c = a.mul(b)          # [4, 10, 18]
c = a * b

# 除法
c = a.div(b)          # [0.25, 0.4, 0.5]
c = a / b

# 标量运算
c = a.add_scalar(10)  # [11, 12, 13]
c = a.mul_scalar(2)   # [2, 4, 6]
```

### 矩阵运算

```python
from tinytorch import Tensor

# 矩阵乘法
A = Tensor([[1, 2], 
            [3, 4]])
B = Tensor([[5, 6], 
            [7, 8]])
C = A.matmul(B)       # [[19, 22], [43, 50]]

# 转置
A_T = A.transpose()   # [[1, 3], [2, 4]]

# 点积
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
dot = a.dot(b)        # 32 (1*4 + 2*5 + 3*6)
```

## 形状操作

### Reshape（重塑）

```python
from tinytorch import Tensor

# 重塑形状
t = Tensor([1, 2, 3, 4, 5, 6])
t2 = t.reshape((2, 3))
print(t2.shape.dims)  # (2, 3)

t3 = t.reshape((3, 2))
print(t3.shape.dims)  # (3, 2)

# 注意：元素总数必须相同
# t.reshape((2, 2))  # 错误！6 != 4
```

### Squeeze 和 Unsqueeze

```python
from tinytorch import Tensor

# Squeeze：移除长度为1的维度
t = Tensor([[[1, 2, 3]]])  # (1, 1, 3)
t2 = t.squeeze()           # (3,)

# Unsqueeze：添加长度为1的维度
t = Tensor([1, 2, 3])      # (3,)
t2 = t.unsqueeze(0)        # (1, 3)
t3 = t.unsqueeze(1)        # (3, 1)
```

### Flatten（展平）

```python
from tinytorch import Tensor

# 展平为一维
t = Tensor([[1, 2], [3, 4]])
t_flat = t.flatten()
print(t_flat.shape.dims)  # (4,)
print(t_flat.data)        # [1, 2, 3, 4]
```

## 索引和切片

### 基本索引

```python
from tinytorch import Tensor

t = Tensor([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])

# 获取单个元素
val = t.get((1, 2))  # 6

# 设置单个元素
t.set((1, 2), 10)    # [[1, 2, 3], [4, 5, 10], [7, 8, 9]]
```

### 切片操作

```python
from tinytorch import Tensor

t = Tensor([[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]])

# 行切片
row1 = t.slice(0, 1, 2)  # 第1-2行

# 列切片
col2 = t.slice(1, 2, 3)  # 第2-3列
```

## 归约运算

### Sum（求和）

```python
from tinytorch import Tensor

t = Tensor([[1, 2, 3],
            [4, 5, 6]])

# 全部求和
total = t.sum()       # 21

# 按维度求和
sum_row = t.sum(0)    # [5, 7, 9]
sum_col = t.sum(1)    # [6, 15]
```

### Mean（平均值）

```python
from tinytorch import Tensor

t = Tensor([[1, 2, 3],
            [4, 5, 6]])

# 全部平均
mean_all = t.mean()   # 3.5

# 按维度平均
mean_row = t.mean(0)  # [2.5, 3.5, 4.5]
mean_col = t.mean(1)  # [2, 5]
```

### Max/Min（最大值/最小值）

```python
from tinytorch import Tensor

t = Tensor([[1, 5, 3],
            [4, 2, 6]])

# 最大值
max_val = t.max()     # 6
max_row = t.max(0)    # [4, 5, 6]

# 最小值
min_val = t.min()     # 1
min_row = t.min(0)    # [1, 2, 3]
```

## 广播机制

广播（Broadcasting）允许不同形状的张量进行运算。

### 广播规则

1. 从右向左对齐维度
2. 维度不同时，较小的维度扩展为1
3. 维度为1的可以扩展到任意大小

```python
from tinytorch import Tensor

# 示例1：标量广播
a = Tensor([[1, 2, 3],
            [4, 5, 6]])  # (2, 3)
b = Tensor([10])         # (1,)
c = a.add(b)             # (2, 3) -> [[11, 12, 13], [14, 15, 16]]

# 示例2：向量广播
a = Tensor([[1, 2, 3],
            [4, 5, 6]])  # (2, 3)
b = Tensor([10, 20, 30]) # (3,)
c = a.add(b)             # [[11, 22, 33], [14, 25, 36]]

# 示例3：矩阵广播
a = Tensor([[1], [2]])   # (2, 1)
b = Tensor([[10, 20]])   # (1, 2)
c = a.add(b)             # (2, 2) -> [[11, 21], [12, 22]]
```

## 数学函数

### 指数和对数

```python
from tinytorch import Tensor

t = Tensor([1, 2, 3])

# 指数函数
exp_t = t.exp()       # [e^1, e^2, e^3]

# 对数函数
log_t = t.log()       # [ln(1), ln(2), ln(3)]

# 幂函数
pow_t = t.pow(2)      # [1, 4, 9]

# 平方根
sqrt_t = t.sqrt()     # [1, 1.414, 1.732]
```

### 激活函数

```python
from tinytorch import Tensor

t = Tensor([-2, -1, 0, 1, 2])

# ReLU
relu_t = t.relu()     # [0, 0, 0, 1, 2]

# Sigmoid
sigmoid_t = t.sigmoid()  # [0.119, 0.268, 0.5, 0.731, 0.881]

# Tanh
tanh_t = t.tanh()     # [-0.964, -0.762, 0, 0.762, 0.964]
```

## 实践示例：图像处理

```python
from tinytorch import Tensor

# 创建一个简单的 3x3 图像
image = Tensor([[0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]])

# 创建一个卷积核（边缘检测）
kernel = Tensor([[-1, -1, -1],
                 [-1,  8, -1],
                 [-1, -1, -1]])

# 归一化
mean = image.mean()
std_dev = ((image.sub_scalar(mean)).pow(2)).mean().sqrt()
normalized = (image.sub_scalar(mean)).div_scalar(std_dev)

print("归一化后的图像:", normalized.data)
```

## 实践示例：批量数据处理

```python
from tinytorch import Tensor

# 批量数据 (batch_size=2, features=3)
batch = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])

# 批量归一化
batch_mean = batch.mean(0)  # 每个特征的均值
batch -= batch_mean         # 中心化

batch_std = batch.pow(2).mean(0).sqrt()
batch /= batch_std          # 标准化

print("批量归一化后:", batch.data)
```

## 性能提示

1. **避免频繁创建小张量**：尽量批量处理
2. **使用就地操作**：减少内存分配
3. **注意广播开销**：大规模广播会消耗内存
4. **选择合适的数据类型**：float32 vs int32

## 下一步

- [03_autograd_system.md](03_autograd_system.md) - 学习自动微分系统
- [04_building_networks.md](04_building_networks.md) - 构建神经网络

## 练习题

1. 创建一个 5x5 的随机张量，计算其转置并验证 (A^T)^T = A
2. 实现批量矩阵乘法：(batch_size, m, k) × (batch_size, k, n)
3. 使用张量操作实现 softmax 函数
4. 创建一个函数，将图像张量从 (H, W, C) 转换为 (C, H, W)
