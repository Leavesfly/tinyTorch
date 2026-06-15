# 09 · API 参考文档

本文档是 tinyTorch **公共 API 的速查手册**，按模块分组，按"类 / 函数"粒度列出签名、关键参数、返回值与错误抛出。更详尽的原理说明请参见对应模块文档（04 ~ 08）。

> **约定**：
> - 所有带 `_` 前缀的符号都是私有实现细节，不在此处列出。
> - 所有"构造函数"按 Python `__init__` 签名书写（省略 `self`）。
> - "返回值"栏描述正常路径的返回；"抛出"栏列出显式 `raise` 的异常。

## 9.1 顶层导入（`tinytorch`）

```python
from tinytorch import NdArray, Shape, Tensor, Function, no_grad
from tinytorch import ndarr, autograd, nn, ml, rl, utils, constants
```

| 名称 | 类型 | 说明 |
|------|------|------|
| `NdArray` | class | 多维数组，见 9.2 |
| `Shape` | class | 形状描述，见 9.2 |
| `Tensor` | class | 带自动微分的张量，见 9.3 |
| `Function` | class | 可微分算子基类，见 9.3 |
| `no_grad` | context manager | 关闭梯度追踪，见 9.3 |
| `__version__` | str | 版本号 |

## 9.2 `tinytorch.ndarr`

### 9.2.1 `NdArray`

#### 构造

```python
NdArray(data, shape=None, dtype='float32')
```

| 参数 | 说明 |
|------|------|
| `data` | Python 数值 / 扁平列表 / 嵌套列表 |
| `shape` | `Shape` 或元组。**嵌套列表时该参数会被忽略**（形状完全由嵌套结构推断）；扁平列表时若省略则默认 `(len(data),)`，若提供则会校验 `len(data) == shape.size`，不一致抛 `ValueError`；标量时若省略默认 `(1,)` |
| `dtype` | `'float32'` 或 `'int32'`，其他值抛 `ValueError` |

#### 工厂方法（classmethod）

| 签名 | 说明 |
|------|------|
| `NdArray.zeros(shape, dtype='float32')` | 全 0 |
| `NdArray.ones(shape, dtype='float32')` | 全 1 |
| `NdArray.randn(shape, seed=None, dtype='float32')` | 标准正态（Box-Muller） |
| `NdArray.uniform(low, high, shape, seed=None, dtype='float32')` | `U(low, high)` |

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `data` | `list` | 扁平存储（行优先） |
| `shape` | `Shape` | 形状对象 |
| `dtype` | `str` | `'float32'` / `'int32'` |

#### 方法

**逐元素运算**（均支持标量 + 广播）：

| 方法 | 等价运算符 |
|------|------------|
| `add(other)` | `a + b` |
| `sub(other)` | `a - b` |
| `mul(other)` | `a * b` |
| `div(other)` | `a / b` |
| `neg()` | `-a` |

**矩阵运算**：

| 方法 | 说明 |
|------|------|
| `matmul(other)` | 2D 矩阵乘，等价 `a @ b`；高维抛 `NotImplementedError` |
| `transpose(axes=None)` | 维度置换；`axes=None` 反转所有维度 |
| `reshape(new_shape)` | 支持一个 `-1` 自动推断 |

**归约运算**：

| 方法 | 签名 |
|------|------|
| `sum(axis=None, keepdims=False)` | 求和 |
| `mean(axis=None, keepdims=False)` | 均值 |
| `max(axis=None, keepdims=False)` | 最大值 |
| `min(axis=None, keepdims=False)` | 最小值 |

**数学函数 / 激活**：

| 方法 | 说明 |
|------|------|
| `exp()` | `e^x`，预钳位避免溢出 |
| `log()` | `ln(x)`，`x=0` → `-inf`，`x<0` → `nan` |
| `sqrt()` | `sqrt(x)`，`x<0` → `nan` |
| `pow(exponent)` | `x ** exponent` |
| `relu()` / `sigmoid()` / `tanh()` | 对应激活函数 |

**辅助**：

| 方法 | 说明 |
|------|------|
| `to_list()` | 按 `shape` 还原为嵌套 Python 列表 |
| `copy()` | 深拷贝 |
| `__repr__()` | 字符串表示 |

### 9.2.2 `Shape`

```python
Shape(dims)
```

| 参数 | 说明 |
|------|------|
| `dims` | 维度元组，每维必须 `> 0`，否则抛 `ValueError` |

**属性**：

| 属性 | 说明 |
|------|------|
| `dims` | 维度元组，如 `(2, 3, 4)` |
| `ndim` | 维度数 |
| `size` | 元素总数 |
| `strides` | 行优先步长，如 `(12, 4, 1)` |

**方法**：

| 方法 | 说明 |
|------|------|
| `linear_index(indices)` | 多维索引 → 扁平索引 |
| `can_broadcast(other)` | 是否可广播 |
| `broadcast_with(other)` | 返回广播后的 `Shape`；不兼容抛 `ValueError` |
| `reshape(new_dims)` | 保持元素总数的新 `Shape` |
| `transpose(axes=None)` | 维度置换 |

## 9.3 `tinytorch.autograd`

### 9.3.1 `Tensor`

#### 构造

```python
Tensor(value, name=None, requires_grad=True)
```

| 参数 | 说明 |
|------|------|
| `value` | 必须是 `NdArray`，否则抛 `TypeError` |
| `name` | 节点名，省略时自动命名 |
| `requires_grad` | 是否追踪梯度 |

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `value` | `NdArray` | 存储的数值 |
| `grad` | `Optional[NdArray]` | 反传后累积的梯度；初始 `None` |
| `creator` | `Optional[Function]` | 产生此 Tensor 的算子 |
| `requires_grad` | `bool` | 是否需要梯度 |
| `name` | `str` | 名称 |
| `shape` | `Shape` | 代理 `value.shape` |

#### 方法

**反传**：

| 方法 | 说明 |
|------|------|
| `backward(grad_output=None, retain_graph=False)` | 反向传播。非标量输出必须显式传 `grad_output`，否则抛 `ValueError` |
| `clear_grad()` | 置 `grad = None` |
| `detach()` | 返回数据拷贝且 `requires_grad=False` 的新 Tensor |
| `unchain_backward()` | 清空计算图的 `creator` 链 |

**运算方法**（构建计算图）：

| 方法 | 等价运算符 | 备注 |
|------|------------|------|
| `add(other)` | `+` | 标量/Tensor 混算自动包装 |
| `sub(other)` | `-` | |
| `mul(other)` | `*` | |
| `div(other)` | `/` | |
| `neg()` | `-a` | |
| `pow(exponent)` | `a ** k` | |
| `matmul(other)` | `a @ b` | 仅 2D |
| `exp()` / `log()` / `sqrt()` | — | |
| `transpose(axes=None)` | — | |
| `reshape(new_shape)` | — | |
| `sum(axis=None, keepdims=False)` | — | |
| `mean(axis=None, keepdims=False)` | — | |
| `relu()` / `sigmoid()` / `tanh()` | — | |
| `leaky_relu(negative_slope=0.01)` | — | |

### 9.3.2 `Function`

```python
class MyOp(Function):
    def forward(self, *inputs):       # inputs 均为 NdArray
        ...                           # 返回 NdArray 或 list/tuple of NdArray
    def backward(self, grad_output):  # 返回 List[NdArray]
        ...
```

#### 核心方法

| 方法 | 说明 |
|------|------|
| `call(*inputs)` / `__call__(*inputs)` | 触发前向，构建计算图。多输出时返回 `tuple[Tensor]` |
| `forward(*inputs)` | **子类必须实现** |
| `backward(grad_output)` | **子类必须实现**，返回"每个 input 的梯度"的列表 |
| `save_for_backward(*tensors)` | 保存反传需要的 `NdArray`（仅在 `forward` 内调用） |
| `get_saved_tensors()` | 取回保存的张量 |
| `clear_saved_tensors()` | 清空（框架在 `backward` 后自动调用） |

### 9.3.3 `no_grad`

```python
with no_grad():
    ...
```

上下文内创建的所有 `Tensor` 都不会构建计算图（`creator=None`）。通过 `Tensor._grad_enabled` 类变量实现，可**嵌套**，退出时恢复之前状态。

### 9.3.4 计算图可视化

全部位于 `tinytorch.autograd.graph_viz`：

| 函数 | 签名 |
|------|------|
| `extract_graph` | `extract_graph(output_tensor, max_depth=100) -> Dict[str, List]` |
| `extract_module_graph` | `extract_module_graph(module) -> Dict[str, List]` |
| `visualize_graph` | `visualize_graph(output_tensor=None, port=8098, auto_open=True, module=None) -> HTTPServer` |
| `export_graph_html` | `export_graph_html(output_tensor, file_path, module=None) -> None` |

## 9.4 `tinytorch.nn`

### 9.4.1 `Module`

| 方法 | 签名 / 说明 |
|------|-------------|
| `__init__(name=None)` | 省略 `name` 时用类名 |
| `forward(*inputs, **kwargs)` | **抽象方法** |
| `__call__(*inputs, **kwargs)` | 调用 `forward` |
| `parameters(recursive: bool = True) -> List[Parameter]` | 参数列表；`recursive=False` 时只返回本模块直接注册的参数 |
| `named_parameters(prefix: str = '', recursive: bool = True) -> Iterator[(str, Parameter)]` | 同上，附带完整路径名 |
| `named_buffers(prefix: str = '', recursive: bool = True) -> Iterator[(str, Tensor)]` | 缓冲区及其完整路径名 |
| `modules() / named_modules(prefix: str = '')` | 递归遍历所有子模块（含 `self`） |
| `train(mode=True) -> self` | 切换模式，递归 |
| `eval() -> self` | = `train(False)` |
| `zero_grad()` | 清空所有参数 `grad` |
| `register_module(name, module)` | 传 `None` 相当于删除 |
| `register_parameter(name, param)` | 同上 |
| `register_buffer(name, tensor)` | 同上 |
| `state_dict() -> Dict[str, Dict]` | 扁平状态字典 |
| `load_state_dict(state_dict, strict=True) -> Dict[str, List[str]]` | 严格模式缺失/多余键抛 `KeyError` |
| `to_dict() -> Dict` | 含 `class / name / training / state_dict` |

### 9.4.2 `Parameter`

```python
Parameter(value, name=None)
```

| 参数 | 说明 |
|------|------|
| `value` | `NdArray`，参数初始值 |
| `name` | 参数名（可选） |

`Tensor` 的子类，构造时**固定**把 `requires_grad` 传为 `True`（不接受用户覆盖），赋给 `Module` 属性时自动注册到 `_parameters`。

额外方法：

| 方法 | 说明 |
|------|------|
| `to_dict() -> Dict` | 返回 `{name, value, shape, dtype, requires_grad}` |
| `Parameter.from_dict(data) -> Parameter` | 从字典还原参数（静态方法） |

### 9.4.3 容器

| 类 | 签名 | 关键方法 |
|----|------|----------|
| `Sequential` | `Sequential(*layers, name=None)` | `add(module, name=None) -> Sequential`（支持链式）、`__len__`、`__getitem__(idx: int)`（**仅 int**，不支持 slice）、`__iter__`、`layers` 属性 |
| `ModuleList` | `ModuleList(modules=None, name=None)` | `append(module) -> ModuleList`、`__len__`、`__getitem__(idx: int)`、`__iter__`；**调用 `forward` 会抛 `NotImplementedError`** |

两者 `add` / `append` 在传入的 `module` 不是 `Module` 实例时会抛 `TypeError`。

### 9.4.4 `nn.init`

| 函数 | 签名 |
|------|------|
| `uniform(a, b, shape, dtype='float32')` | 工厂函数，返回新 `NdArray` |
| `uniform_(tensor, a=0.0, b=1.0)` | 原地 |
| `normal_(tensor, mean=0.0, std=1.0)` | 原地 |
| `constant_(tensor, val)` | 原地 |
| `zeros_(tensor)` / `ones_(tensor)` | 原地 |
| `xavier_uniform_(tensor, gain=1.0)` | Glorot |
| `xavier_normal_(tensor, gain=1.0)` | Glorot |
| `kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')` | He |
| `kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')` | He |
| `calculate_gain(nonlinearity='linear')` | 返回增益因子 |

**别名**：`glorot_uniform_` / `glorot_normal_` / `he_uniform_` / `he_normal_` 分别等于对应的 `xavier_*` / `kaiming_*`。

### 9.4.5 层一览

| 类 | 构造签名 | 输入形状 → 输出形状 |
|----|----------|---------------------|
| `Linear` | `Linear(in_features, out_features, use_bias=True, name=None)` | `(N, in)` / `(N, T, in)` → 同形换 `in` 为 `out` |
| `ReLU` / `Sigmoid` / `Tanh` | `Class()` | 与输入同形 |
| `LeakyReLU` | `LeakyReLU(negative_slope=0.01)` | 与输入同形 |
| `LayerNorm` | `LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, name=None)` | 与输入同形，末尾若干维必须匹配 `normalized_shape` |
| `Dropout` | `Dropout(p=0.5, name=None)`，`p ∈ [0, 1)` | 与输入同形，`eval` 时恒等 |
| `Embedding` | `Embedding(num_embeddings, embedding_dim, padding_idx=None, name=None)` | `(..., L)` 整数索引 → `(..., L, embedding_dim)` |
| `Conv2d` | `Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, use_bias=True)` | `(N, C, H, W)` → `(N, out_C, H', W')`；均为 `int` |
| `RNN` | `RNN(input_size, hidden_size, use_bias=True)` | `(N, T, in)` → `(N, T, hidden)` |
| `LSTM` | `LSTM(input_size, hidden_size, use_bias=True)` | `(N, T, in)` → `(h_all, c_final)` |
| `GRU` | `GRU(input_size, hidden_size, use_bias=True)` | `(N, T, in)` → `(N, T, hidden)` |
| `MultiHeadAttention` | `MultiHeadAttention(embed_dim, num_heads, dropout=0.0)` | `forward(query, key=None, value=None, mask=None)`；`embed_dim` 必须能被 `num_heads` 整除 |

循环层 `forward(input, initial_states=None)`：`RNN`/`GRU` 的 `initial_states` 是单 Tensor `(N, hidden)`，`LSTM` 是 `(h_0, c_0)` 元组。

## 9.5 `tinytorch.ml`

### 9.5.1 `Model`

| 方法 | 签名 |
|------|------|
| `__init__(name, module)` | — |
| `forward(x) -> Tensor` / `__call__(x)` | 转发到 `module` |
| `parameters() / named_parameters()` | 转发 |
| `train() / eval() / zero_grad()` | 转发 |
| `save(file_path)` | `pickle`，⚠️ 不要加载不可信文件 |
| `Model.load(file_path, module=None) -> Model` | **必须**传 `module`，否则抛 `ValueError` |
| `save_parameters(file_path)` / `load_parameters(file_path)` | 仅参数，非严格加载 |

### 9.5.2 `Optimizer` / `SGD` / `Adam`

| 类 | 构造签名 |
|----|----------|
| `Optimizer` | `Optimizer(params, learning_rate=0.01, **kwargs)`；`kwargs` 支持 `lr` 别名 |
| `SGD` | `SGD(params, learning_rate=0.01, momentum=0.0, weight_decay=0.0)` |
| `Adam` | `Adam(params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)`；支持 `betas=(b1, b2)` 元组 |

**公共方法**：

| 方法 | 说明 |
|------|------|
| `step()` | 执行一步更新；抽象方法，基类调用抛 `NotImplementedError` |
| `zero_grad()` | 遍历参数 `clear_grad` |
| `lr` / `learning_rate` | 可读写属性 |
| `state_dict() -> Dict` | `{'learning_rate', 'state'}` |
| `load_state_dict(state_dict)` | — |

### 9.5.3 `Loss` / `MSELoss` / `CrossEntropyLoss` / `BCELoss`

| 类 | 构造签名 | 输入要求 |
|----|----------|----------|
| `Loss` | `Loss(name=None, reduction='mean')` | `reduction ∈ {'mean','sum','none'}` |
| `MSELoss` | `MSELoss(reduction='mean')` | `pred` 与 `target` 同形 |
| `CrossEntropyLoss` | `CrossEntropyLoss(reduction='mean')` | `logits(N, C)` + `target(N,)` 类别索引 |
| `BCELoss` | `BCELoss(reduction='mean')` | `input` 必须已经过 Sigmoid |

所有损失类都支持 `loss_fn(pred, target) -> Tensor`。

### 9.5.4 `Evaluator` 家族

| 类 | 主方法 | 说明 |
|----|--------|------|
| `Evaluator` | `evaluate(predictions, targets) -> float` | 基类，抽象 |
| `AccuracyEvaluator` | `evaluate(predictions, targets)` | `predictions` 可以是概率向量或类别索引 |
| `PrecisionRecallEvaluator` | `evaluate(...)` 返 F1；`evaluate_all(...)` 返 `{precision, recall, f1}` | 构造参数 `average='binary'` |
| `RegressionEvaluator` | `evaluate(...)` 返 MSE；`evaluate_all(...)` 返 `{mae, mse, rmse}` | — |

### 9.5.5 `DataSet`

```python
DataSet(data, labels, batch_size=32, shuffle=True)
```

| 方法 | 说明 |
|------|------|
| `__len__()` / `__getitem__(i)` | 单样本访问 |
| `iter_batches() -> Iterator[(NdArray, NdArray)]` | 惰性批次迭代 |
| `get_batches() -> List[(NdArray, NdArray)]` | 全部批次 |
| `__iter__()` | 产出**原始列表**批次（非 `NdArray`） |
| `shuffle_data()` | 打乱内部索引 |
| `split(ratio) -> (DataSet, DataSet)` | `ratio ∈ (0, 1)` |

### 9.5.6 `Trainer`

```python
Trainer(model, dataset, optimizer, loss_fn,
        max_epochs=10, print_interval=10,
        val_dataset=None, visualizer=None)
```

| 方法 | 说明 |
|------|------|
| `train()` | 完整训练循环 |
| `train_epoch(epoch) -> float` | 单 epoch 平均损失 |
| `validate() -> float` | 验证集平均损失；未提供 `val_dataset` 抛 `ValueError` |
| `save_checkpoint(file_path)` / `load_checkpoint(file_path)` | ⚠️ 不回填模型权重 |

### 9.5.7 `Monitor` / `EarlyStopping`

| 类 | 构造签名 | 主方法 |
|----|----------|--------|
| `Monitor` | `Monitor()` | `start()`、`record(name, value, epoch=None)`、`record_epoch(epoch, metrics)`、`get_metric(name)`、`get_latest(name)`、`get_best(name, mode='min'\|'max')`、`print_epoch / print_summary`、`save_history / load_history`（pickle） |
| `EarlyStopping` | `EarlyStopping(patience=10, mode='min', min_delta=0.0)` | `step(score) -> bool`、`reset()`、属性 `should_stop` |

### 9.5.8 `TrainingVisualizer`

```python
TrainingVisualizer(port=8097, auto_open=True)
```

| 方法 | 说明 |
|------|------|
| `start_server()` / `stop_server()` | 后台 HTTP 服务 |
| `begin_training()` | 状态置 `'running'`，记录起始时间 |
| `record_epoch(epoch, train_loss=None, val_loss=None, **extra_metrics)` | 追加一轮数据 |
| `finalize()` | 状态置 `'done'`（服务不关） |
| `set_graph(output_tensor=None, module=None)` | 设置要展示的计算图 |
| `get_data_snapshot() -> Dict` | 训练数据快照（线程安全） |
| `get_graph_snapshot() -> Dict` | 计算图快照 |
| `save_data(file_path)` / `load_data(file_path)` | JSON |
| `export_html(file_path)` | 内嵌数据的单文件 HTML |

## 9.6 `tinytorch.utils`

### 9.6.1 `tinytorch.utils.data`

| 符号 | 签名 / 说明 |
|------|-------------|
| `Dataset` | 抽象基类；需实现 `__len__` 与 `__getitem__` |
| `IterableDataset` | 抽象基类；需实现 `__iter__` |
| `Sampler` | 抽象基类 |
| `SequentialSampler(dataset)` | 顺序索引 |
| `RandomSampler(dataset)` | 无放回随机 |
| `BatchSampler(sampler, batch_size, drop_last)` | 批索引；`batch_size <= 0` 抛 `ValueError` |
| `default_collate(batch)` | 默认合并函数，见 08.2.4 |
| `DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, drop_last=False, collate_fn=default_collate, num_workers=0, pin_memory=False)` | 参数见 08.2.3；`num_workers != 0` 或 `pin_memory=True` 抛 `NotImplementedError` |

### 9.6.2 `tinytorch.utils.random`

| 函数 | 说明 |
|------|------|
| `seed(value)` | 设置全局 RNG 种子 |
| `generator(seed_value=None) -> random.Random` | 独立 RNG |
| `random() -> float` | `[0, 1)` 均匀 |
| `uniform(a, b) -> float` | `[a, b]` 均匀 |
| `gauss(mean, std) -> float` | 高斯 |
| `shuffle(values)` | 原地打乱 |

## 9.7 `tinytorch.constants`

| 常量 | 值 | 用途 |
|------|-----|------|
| `DEFAULT_EPSILON` | `1e-10` | 反传防除零 |
| `EXP_OVERFLOW_THRESHOLD` | `709.0` | `exp` 上溢钳位 |
| `EXP_UNDERFLOW_THRESHOLD` | `-745.0` | `exp` 下溢钳位 |
| `BOX_MULLER_MIN_U1` | `1e-10` | `randn` 用的 Box-Muller 截断 |

> ⚠️ 这些常量被框架代码标注为**内部实现细节**，不要在业务代码中依赖它们的具体值。

## 9.8 `tinytorch.rl`

### 9.8.1 空间与环境

| API | 说明 |
|-----|------|
| `Discrete(n)` | 离散空间，取值为 `0 ~ n-1` |
| `Box(low, high, shape)` | 一维连续向量空间 |
| `MultiDiscrete(nvec)` | 多离散空间 |
| `Env` | 环境基类，约定 `reset()` / `step(action)` |
| `VectorEnv(env_fns)` | 顺序管理多个环境实例 |
| `GridWorldEnv(width=4, height=4, start=(0, 0), goal=None, obstacles=None, step_reward=-0.01, goal_reward=1.0, wall_reward=-0.1, max_steps=None)` | 内置二维网格世界 |

### 9.8.2 经验与轨迹

| API | 说明 |
|-----|------|
| `Transition(state, action, reward, next_state, done)` | 单步交互样本 |
| `ReplayBuffer(capacity=10000)` | 固定容量经验回放 |
| `PrioritizedReplayBuffer(capacity=10000, alpha=0.6, beta=0.4, epsilon=1e-6)` | 优先级经验回放 |
| `RolloutBuffer()` | on-policy 轨迹缓存，支持 return / GAE |

### 9.8.3 Agent

| API | 说明 |
|-----|------|
| `QLearningAgent(action_size, learning_rate=0.1, gamma=0.99, epsilon=0.1, epsilon_min=0.01, epsilon_decay=1.0)` | 表格 Q-learning |
| `DQNAgent(q_network, optimizer, state_dim, action_size, gamma=0.99, epsilon=0.1, epsilon_min=0.01, epsilon_decay=1.0, replay_buffer=None, target_network=None, batch_size=32)` | 深度 Q 网络 |
| `DoubleDQNAgent(...)` | Double DQN，签名同 `DQNAgent` |
| `PolicyGradientAgent(policy_network, optimizer, state_dim, action_size, gamma=0.99, normalize_returns=True)` | REINFORCE |
| `ActorCriticAgent(policy_network, value_network, policy_optimizer, value_optimizer, state_dim, action_size, gamma=0.99, normalize_advantages=True)` | Advantage Actor-Critic |
| `PPOAgent(policy_network, value_network, policy_optimizer, value_optimizer, state_dim, action_size, gamma=0.99, clip_epsilon=0.2, normalize_advantages=True)` | 简化 PPO |

更多说明见 [12 · 强化学习（rl）模块](./12-强化学习rl模块.md)。

## 9.9 异常速查

| 抛出位置 | 异常类型 | 典型触发条件 |
|----------|----------|--------------|
| `NdArray.__init__` | `ValueError` | 不合法 `dtype`；参差嵌套（`ragged dimensions are not allowed`）；扁平列表时 `len(data) != shape.size` |
| `NdArray.__init__` | `TypeError` | 传入的 `data` 既不是标量也不是 `list` |
| `Shape.__init__` | `ValueError` | 某维 `<= 0` |
| `Linear.forward` | `ValueError` | 输入既非 2D 也非 3D |
| `Shape.broadcast_with` | `ValueError` | 不兼容的 shape |
| `NdArray.matmul` | `NotImplementedError` | 维度 ≠ 2 |
| `Tensor.__init__` | `TypeError` | `value` 不是 `NdArray` |
| `Tensor.backward` | `ValueError` | 非标量输出未传 `grad_output`，或 shape 不匹配 |
| `Module.load_state_dict` | `KeyError` | `strict=True` 时出现缺失 / 多余键 |
| `BatchSampler.__init__` | `ValueError` | `batch_size <= 0` |
| `DataLoader.__init__` | `NotImplementedError` | `num_workers != 0` 或 `pin_memory=True` |
| `DataLoader.__init__` | `ValueError` | `batch_sampler` 与其他参数冲突；`IterableDataset` + `shuffle/sampler` |
| `DataSet.split` | `ValueError` | `ratio` 不在 `(0, 1)` |
| `Monitor.get_best` | `ValueError` | `mode` 不是 `'min'` / `'max'` |
| `Loss.__init__` | `ValueError` | `reduction` 不在 `{'mean','sum','none'}` |
| `LayerNorm.forward` | `ValueError` | 输入末尾维度与 `normalized_shape` 不一致 |
| `Conv2d.forward` | `ValueError` | 输入非 4D 或 `in_channels` 不匹配 |
| `MultiHeadAttention.__init__` | `ValueError` | `embed_dim % num_heads != 0` |
| `Dropout.__init__` | `ValueError` | `p` 不在 `[0, 1)` |
| `Embedding.__init__` | `ValueError` | `padding_idx` 越界 |
| `Model.load` | `ValueError` | 未传入 `module` |
| `Optimizer.step` | `NotImplementedError` | 基类直接调用 |
| `Function.forward / backward` | `NotImplementedError` | 子类未实现 |

## 9.10 常用导入清单

```python
# 核心数值 / 自动微分
from tinytorch import NdArray, Shape, Tensor, Function, no_grad

# 神经网络
from tinytorch.nn import (
    Module, Parameter,
    Sequential, ModuleList,
    Linear, Conv2d, Embedding,
    RNN, LSTM, GRU, MultiHeadAttention,
    LayerNorm, Dropout,
    ReLU, Sigmoid, Tanh, LeakyReLU,
)
from tinytorch.nn import init

# 训练框架
from tinytorch.ml import (
    Model, Trainer, DataSet,
    SGD, Adam,
    MSELoss, CrossEntropyLoss, BCELoss,
    AccuracyEvaluator, PrecisionRecallEvaluator, RegressionEvaluator,
    Monitor, EarlyStopping, TrainingVisualizer,
)

# 数据加载
from tinytorch.utils import Dataset, IterableDataset, DataLoader, default_collate
from tinytorch.utils import random as tt_random

# 强化学习
from tinytorch.rl import (
    GridWorldEnv, VectorEnv,
    ReplayBuffer, PrioritizedReplayBuffer, RolloutBuffer,
    QLearningAgent, DQNAgent, DoubleDQNAgent, PPOAgent,
)

# 计算图可视化
from tinytorch.autograd import export_graph_html, visualize_graph
```

---

**上一篇** ← [08 · 工具与数据加载（utils）模块](./08-工具与数据加载utils模块.md)  ·  **下一篇** → [10 · 常见问题解答（FAQ）](./10-常见问题解答FAQ.md)
