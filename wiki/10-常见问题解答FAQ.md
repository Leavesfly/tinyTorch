# 10 · 常见问题解答（FAQ）

本文档收集使用 tinyTorch 过程中的高频疑问，涵盖安装 / 张量与自动微分 / 神经网络 / 训练 / 数据加载 / 性能 / 调试等方向。每个问题都给出**简短答案 + 指向源码或其他文档**的佐证。

## 11.1 安装与环境

### Q1：最低需要什么 Python 版本？有哪些依赖？

**答**：Python 3.7+，**没有任何运行时依赖**。`requirements.txt` 仅列出运行时（空），`pyproject.toml` 的 `dependencies` 字段也是空的。全部功能都用标准库 + 纯 Python 实现。

详见 [02 · 安装指南](./02-安装指南.md)。

### Q2：安装后 `import tinytorch` 很慢 / 报错怎么办？

**答**：

- `import tinytorch` 会级联导入 `ndarr / autograd / nn / ml / utils / constants`，首次导入约数百毫秒属正常；
- 如果报 `ModuleNotFoundError: No module named 'tinytorch'`，检查当前 Python 是否是安装时用的同一个（`python -c "import sys; print(sys.executable)"`）。

### Q3：如何验证安装成功？

**答**：

```bash
python -c "from tinytorch import NdArray, Tensor; print('ok')"
python run_tests.py          # 运行全部单元测试
```

## 11.2 张量与自动微分

### Q4：为什么 `Tensor([1.0, 2.0])` 会报 `TypeError`？

**答**：`Tensor.__init__` 对 `value` 做了严格类型检查，必须传 `NdArray`。正确写法：

```python
from tinytorch import NdArray, Tensor
x = Tensor(NdArray([1.0, 2.0]))
```

这是刻意的设计：让 `Tensor` 的"value 总是 `NdArray`"不变式成立，避免每个算子都要做一遍类型转换。

### Q5：`loss.backward()` 报 `ValueError: grad_output must be provided for non-scalar tensor`？

**答**：`backward()` 只能对**标量**（shape 为 `(1,)`）直接调用。非标量输出必须传 `grad_output`：

```python
# 方案 1：先归约成标量
loss = squared_error.mean()
loss.backward()

# 方案 2：显式传同形的 grad_output
from tinytorch import NdArray
y.backward(grad_output=NdArray.ones(y.shape))
```

详见 [05 · Autograd 模块 §5.4.2](./05-Autograd模块.md)。

### Q6：为什么每步训练前要 `optimizer.zero_grad()`？

**答**：tinyTorch 对同名参数的梯度**自动累加**（这是支持菱形依赖、双塔网络等结构所必需的）。不清零就意味着"上一步的梯度 + 本步梯度"一起用来更新，行为等同于变相扩大 batch size 且不可控。

源码证据：`Tensor._propagate_gradients` 中 `inp.grad = g if inp.grad is None else inp.grad.add(g)`。

### Q7：`retain_graph=True` 什么时候用？

**答**：默认 `backward()` 会调用 `unchain_backward()` 断开整张图的 `creator`，从而释放内存。下列场景需要显式保留：

- 对同一张图多次 `backward()`（如计算多个 loss 分别的梯度）；
- `backward()` 之后还要用 `extract_graph` / `visualize_graph` 可视化；
- 某些高阶梯度场景（本框架未内置，但第三方扩展可能用到）。

### Q8：`detach()` 和 `no_grad()` 有什么区别？

**答**：

| 项 | `detach()` | `with no_grad():` |
|----|-----------|---------------------|
| 作用范围 | 单个 Tensor | 块内所有新建的 Tensor |
| 底层机制 | 返回 `requires_grad=False` 的**数据拷贝** | 修改 `Tensor._grad_enabled` 类变量 |
| 是否会割断计算图 | 是（新张量没有 creator） | 是（块内创建的张量 `creator=None`） |
| 原张量是否改变 | 否 | 否（只影响新建的） |

### Q9：自定义 `Function` 时 `forward` 能返回 `Tensor` 吗？

**答**：**不能**。`Function.forward` 必须返回 `NdArray`（或 `NdArray` 的 list / tuple）。框架会在 `Function.call` 里把 `NdArray` 包装成 `Tensor` 并挂 `creator`。误传会导致后续 `unpack_value` 失败。

### Q10：`Function` 里 `save_for_backward` 保存的张量什么时候被清理？

**答**：`backward()` 调用完成后框架自动调用 `clear_saved_tensors()` 清空。因此：

- 同一个 `Function` 实例**不能**跨多次 `backward` 复用 `saved_tensors`；
- 如果需要多次反传，`backward(retain_graph=True)` 之后仍要**保持 `Function.inputs` / `saved_tensors` 的引用**不被回收。

## 11.3 神经网络层

### Q11：`Linear` 能处理 3D 输入（`batch × seq × dim`）吗？

**答**：可以。`Linear.forward` 会自动识别 2D / 3D：3D 时先 `reshape` 到 `(batch*seq, in)` 做矩阵乘，再恢复形状。其他维度会抛 `ValueError`。

### Q12：`Conv2d` 支持非正方形核、不对称 stride/padding 吗？

**答**：**不支持**。`kernel_size` / `stride` / `padding` 都是 `int`，没有元组形式；卷积核形状固定为 `(out_C, in_C, k, k)`。需要时请用两次卷积叠加或用 reshape 绕路。

### Q13：`LSTM` 的 `forward` 返回两个值，`Sequential` 能直接塞进去吗？

**答**：**不能**。`Sequential` 的链式传递假设每一层的输出是下一层的单一输入；`LSTM` 返回 `(h_all, c_final)` 会打断这一约定。建议：

- 自定义一个 `Module`，在 `forward` 里手动拆开 LSTM 的返回值；
- 或用 `ModuleList` + 手写前向。

### Q14：`Dropout` 在评估阶段还会掉数吗？

**答**：**不会**。`Dropout.forward` 会检查 `self.training`：`eval()` 模式（`training=False`）下直接返回输入；训练模式下走反向缩放（inverted dropout）。

### Q15：为什么 `MultiHeadAttention` 构造时抛 `ValueError`？

**答**：因为 `embed_dim % num_heads != 0`。比如 `embed_dim=60, num_heads=8` 会报错，改成 `num_heads=4` 或 `embed_dim=64` 即可。

### Q16：如何冻结某些层的参数？

**答**：把它们的 `requires_grad` 置 `False`：

```python
for p in model.module.fc1.parameters():
    p.requires_grad = False
```

这样这些参数在 `backward` 时不会累积梯度，`optimizer.step()` 也会跳过（因为 `param.grad is None`）。

### Q17：`register_buffer` 和 `register_parameter` 有什么区别？

**答**：

| 项 | `Parameter` | `Buffer` |
|----|-------------|----------|
| 是否参与梯度更新 | 是（`requires_grad=True`） | 否（要手动置 `False`） |
| 是否保存进 `state_dict` | 是（`kind='parameter'`） | 是（`kind='buffer'`） |
| 典型用途 | 权重、偏置 | BN 的 `running_mean`、位置编码、不可学习的常数张量 |

## 11.4 训练与优化

### Q18：`SGD` 和 `Adam` 怎么选？

**答**：

- **`SGD`**：简单、可解释、收敛稳定；对学习率敏感，常需搭配 `momentum=0.9`；
- **`Adam`**：自适应学习率，对初值不敏感、收敛快；在纯 Python 下单步开销略高于 `SGD`。

新手建议从 `Adam(lr=1e-3)` 起步；深度学习论文复现优先 `SGD + momentum`。

### Q19：想使用 PyTorch 风格的 `lr` 参数行不行？

**答**：可以。`Optimizer.__init__` 支持 `lr` 作为 `learning_rate` 的别名：

```python
optim = SGD(model.parameters(), lr=0.01)        # 等价于 learning_rate=0.01
optim.lr = 0.005                                # 动态调整
```

### Q20：`Adam` 能接 `betas=(0.9, 0.999)` 元组吗？

**答**：可以。`Adam.__init__` 会解构 `kwargs['betas']` 并赋给 `beta1` / `beta2`：

```python
Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
```

### Q21：训练中损失是 NaN / inf 怎么办？

**答**：常见原因：

1. **学习率过大** → 调小 `lr`；
2. **没有做数据归一化** → 输入均值接近 0、方差接近 1；
3. **`log(0)` / `sqrt(负)`** → 本框架对这些情况返回 `-inf` / `nan` 不抛异常（见 [04 · §4.6](./04-NdArray模块.md)）。检查你的自定义损失或算子；
4. **梯度爆炸** → 可手动做梯度裁剪：

```python
for p in model.parameters():
    if p.grad is not None:
        # clamp 每个元素到 [-max_norm, max_norm]
        clamped = [max(-1.0, min(1.0, v)) for v in p.grad.data]
        p.grad.data = clamped
```

（框架目前未内置 `clip_grad_norm`，需自行实现。）

### Q22：`Trainer` 能用 `tinytorch.utils.data.DataLoader` 吗？

**答**：**不能直接使用**。`Trainer` 依赖 `DataSet.iter_batches()` 与 `dataset.batch_size` 属性，而 `DataLoader` 没有这些。若需要 `DataLoader`，请自行手写训练循环（参考 07.7.2 的伪代码）。

### Q23：`trainer.load_checkpoint()` 加载后模型权重没恢复？

**答**：这是**已知设计缺口**。`load_checkpoint` 只恢复 `optimizer_state` 与损失历史，**不回填 `model.module`**。恢复权重请另外 `model.load_parameters(...)` 或 `Model.load(..., module=...)`。

相关源码：`Trainer.load_checkpoint`（`tinytorch/ml/trainer.py`）。

### Q24：如何实现早停？

**答**：用 `EarlyStopping`，手动嵌入循环：

```python
from tinytorch.ml import EarlyStopping
es = EarlyStopping(patience=5, mode='min')

for epoch in range(max_epochs):
    train_one_epoch()
    val_loss = validate()
    if es.step(val_loss):
        break
```

`Trainer` 没有内置早停；如需自动触发，可自行子类化 `Trainer` 覆盖 `train()`。

### Q25：`CrossEntropyLoss` 需要在模型末尾加 Softmax 吗？

**答**：**不需要**。`CrossEntropyLoss` 内部融合了 `log_softmax + nll`，直接传**未经过 softmax 的 logits**。提前加 softmax 反而会导致数值不稳定和梯度错误。

## 11.5 数据加载

### Q26：`DataLoader` 设置 `num_workers=4` 报错？

**答**：当前只支持 `num_workers=0`（单进程）。多进程加载尚未实现（见源码 `DataLoader.__init__`：`raise NotImplementedError`）。这是一份**教学/小规模实验取向**的实现。

### Q27：变长序列怎么组 batch？

**答**：默认 `default_collate` 遇到不等长 `Sequence` 会抛 `ValueError`。要自定义 `collate_fn`：

```python
def pad_collate(batch):
    seqs, labels = zip(*batch)
    max_len = max(len(s) for s in seqs)
    padded = [list(s) + [0] * (max_len - len(s)) for s in seqs]
    return NdArray(padded), NdArray(list(labels))

loader = DataLoader(dataset, batch_size=32, collate_fn=pad_collate)
```

### Q28：为什么 `DataLoader` 里 `shuffle=True` 又传 `sampler` 会报错？

**答**：两者互斥。源码里明确：传了 `sampler` 就不能再 `shuffle=True`（因为"是否打乱"已经由 `sampler` 决定）。

### Q29：`IterableDataset` 为什么不允许 `shuffle`？

**答**：`IterableDataset` 是流式数据源，没有 `__len__` 和 `__getitem__`，无法事先构造索引列表来打乱。如果确实需要打乱，请在 `IterableDataset.__iter__` 内部自行实现 shuffle buffer。

## 11.6 性能

### Q30：训练一个 epoch 要几十秒 / 几分钟，正常吗？

**答**：**正常**。tinyTorch 完全用纯 Python 实现（没有 C/CUDA、没有 NumPy），运算是嵌套 for 循环。性能敏感的场景请选用 PyTorch / JAX；tinyTorch 的定位是**教学与原理实验**。

详细取舍见 [03 · 架构设计 §3.10](./03-架构设计.md)。

### Q31：如何加速小数据集的实验？

**答**：几条实用建议：

- **缩小模型规模**：参数量和算子调用次数都是 Python 开销的主因；
- **推理时加 `no_grad()`**：跳过计算图构建；
- **避免过小的 batch**：`batch=1` 时 Python 开销远高于实际计算；
- **避免逐元素 Python 循环**：自定义算子尽量用 `NdArray` 已有的向量化方法；
- **使用惰性迭代** `DataSet.iter_batches()` 代替 `get_batches()`，避免一次性构造大列表。

### Q32：内存占用很快就涨上去？

**答**：

- `Tensor` 默认 `retain_graph=False` 会在 `backward()` 后释放计算图；如果你一直不 `backward` 或总是 `retain_graph=True`，内存只增不减；
- `save_for_backward` 会持有 `NdArray` 引用；若自定义算子保存了过大张量，考虑只保存必要分量；
- Python 是 GC 延迟回收，可手动 `gc.collect()` 观察回收。

## 11.7 调试

### Q33：怎么查看一个 Tensor 的来源（`creator`）？

**答**：直接访问 `tensor.creator` 属性，它是产生这个 `Tensor` 的 `Function` 实例（叶子节点为 `None`）：

```python
y = x * 2 + 1
print(type(y.creator).__name__)    # 'Add'，即 tinytorch.autograd.ops.basic.Add
print(y.creator.inputs)            # [<Tensor>, <Tensor>]：两个输入 Tensor
print(y.creator.inputs[0].creator) # 继续往上游追溯
```

要**递归**看完整图的话，用 `extract_graph(y)` 返回 `{'nodes': [...], 'edges': [...]}` 字典（见 9.3.4）。

### Q34：打印 Tensor 看不到数值？

**答**：`Tensor.__repr__` 只显示 shape / requires_grad 等元信息。要看数据，用：

```python
print(tensor.value)                # NdArray 的 __repr__
print(tensor.value.to_list())      # 嵌套列表
```

### Q35：`state_dict()` 的 key 是什么格式？

**答**：**扁平路径，以 `.` 分隔**。路径中每一段对应一层 `Module` 注册的"名字"。注意 `Sequential` 和 `ModuleList` 对"名字"的生成规则**不同**：

| 容器 | 默认命名（无自定义 `name`） | 示例路径 |
|------|-----------------------------|----------|
| `Sequential(L1, L2)` | `layer_0` / `layer_1` / ...（源码 `container.py:self.add(layer, name=f'layer_{idx}')`） | `seq.layer_0.weight` |
| `ModuleList([L1, L2])` | `"0"` / `"1"` / ...（源码 `container.py:self.register_module(str(idx), module)`） | `ml.0.weight`、`ml.1.weight` |

举一个综合例子：

```python
class Net(Module):
    def __init__(self):
        super().__init__()
        self.backbone = Sequential(Linear(10, 20), ReLU(), Linear(20, 30))
        self.heads = ModuleList([Linear(30, 5), Linear(30, 5)])

# state_dict() 的键形如：
#   backbone.layer_0.weight
#   backbone.layer_0.bias
#   backbone.layer_2.weight    # ReLU 不含参数，所以中间索引被跳过
#   backbone.layer_2.bias
#   heads.0.weight
#   heads.0.bias
#   heads.1.weight
#   heads.1.bias
```

`Sequential.add(module, name='xxx')` 传入显式 `name` 时会使用该名（但会在命名冲突时自动加数字后缀）。

### Q36：可视化 HTML 打开是空白？

**答**：检查：

- 是否在 `backward()` **之前**调用了 `extract_graph` / `set_graph` / `export_graph_html`；默认 `backward()` 会清空计算图，之后再调用只会看到孤立的输出节点（源码注释已特别强调）；
- `visualize_graph` 启动的服务默认监听 `127.0.0.1:8098`，端口占用时会抛异常；
- 浏览器网络面板看 `/api/graph_data` 请求返回是否为空。

## 11.8 进阶

### Q37：我能用 tinyTorch 训练 GPU 模型吗？

**答**：**不能**。tinyTorch 没有 CUDA 后端，所有运算都在 CPU 上用 Python 循环完成。这也是框架自身定位决定的（教学为主）。

### Q38：如何把 PyTorch 的 `state_dict` 迁移过来？

**答**：两端的 `state_dict` 结构**不同**：

- PyTorch：`{name: Tensor}`；
- tinyTorch：`{name: {'kind', 'value', 'shape', 'dtype', 'requires_grad'}}`。

迁移思路：遍历 PyTorch `state_dict`，为每个参数构造 tinyTorch 期望的 `dict`——其中 `value` 必须是**按 shape 还原的嵌套 Python 列表**（与 `NdArray.to_list()` 的产物一致），`shape` 是 **tuple**。这是因为 tinyTorch 的 `Module.load_state_dict` 内部执行 `NdArray(entry['value'], Shape(tuple(entry['shape'])), dtype=...)`，若传扁平列表 + 期望 shape 会被 `NdArray` 拒绝。

```python
# 从 PyTorch 导出
import json
import torch

def tensor_to_nested_list(t: 'torch.Tensor'):
    """把 torch.Tensor 转成与 NdArray.to_list() 对齐的嵌套 Python 列表。"""
    return t.detach().cpu().tolist()

pt_state = torch.load('pt_model.pth')  # 假设是纯 state_dict
export = {
    name: {
        'kind': 'parameter',
        'value': tensor_to_nested_list(tensor),
        'shape': tuple(tensor.shape),
        'dtype': 'float32',
        'requires_grad': True,
    }
    for name, tensor in pt_state.items()
}

# 序列化时把 tuple 转 list（JSON 不支持 tuple；tinyTorch 侧会再转回 tuple）
json.dump(
    {k: {**v, 'shape': list(v['shape'])} for k, v in export.items()},
    open('tt_state.json', 'w'),
)

# tinyTorch 侧
import json
from tinytorch.nn import Sequential, Linear, ReLU
net = Sequential(...)                          # 与 PyTorch 对齐结构
state = json.load(open('tt_state.json'))
net.load_state_dict(state, strict=False)       # 参数名必须对齐
```

⚠️ 注意事项：

- **权重形状**：PyTorch 的 `nn.Linear.weight` 形状是 `(out, in)`，tinyTorch 的 `Linear` 同样是 `(out, in)`，可以直接对齐；但 `Conv2d` 的权重顺序务必逐维核对。
- **`dtype`**：tinyTorch 只支持 `'float32'` / `'int32'` 两种字符串 `dtype`；PyTorch 的 `torch.float32` 等需映射。
- **`kind`**：`Module.load_state_dict` 当前**不**校验 `kind` 字段（只用 `value` / `shape` / `dtype` / `requires_grad`），但为了语义清晰仍建议填写 `'parameter'` 或 `'buffer'`。
- **`name` 对齐**：tinyTorch 的参数路径以 `.` 分隔（例如 `layer_0.weight`），与 PyTorch 的 `Sequential` 生成的 `0.weight` 不同，需手动重命名。

### Q39：能不能写一个自定义优化器？

**答**：可以。继承 `Optimizer` 并实现 `step()`：

```python
from tinytorch.ml.optimizers.optimizer import Optimizer
from tinytorch.ndarr import NdArray

class MyOptim(Optimizer):
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.shape.dims != p.value.shape.dims:
                grad = grad.reshape(p.value.shape.dims)
            p.value = p.value.sub(grad.mul(self.learning_rate))
```

### Q40：我发现了一个 bug / 有改进想法？

**答**：欢迎贡献！请阅读 [11 · 贡献指南](./11-贡献指南.md)，按 fork → branch → commit → PR 的流程提交；提交前请跑通 `python run_tests.py`。

---

**上一篇** ← [09 · API 参考文档](./09-API参考文档.md)  ·  **下一篇** → [11 · 贡献指南](./11-贡献指南.md)
