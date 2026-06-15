# tinyTorch 技术 Wiki

欢迎来到 **tinyTorch** 项目的技术 wiki。这里汇总了从"项目概览"到"如何贡献代码"所需的全部文档，帮助你快速理解和使用这个纯 Python 实现的轻量级深度学习框架。

> tinyTorch 是一个仅依赖 Python 标准库的教学型深度学习框架，提供 `NdArray`、自动微分、神经网络层和训练工具链的完整实现。

## 📚 文档索引

| 编号 | 文档 | 内容简介 |
|------|------|----------|
| 01 | [项目概述](./01-项目概述.md) | 项目背景、设计目标、整体定位与能力矩阵 |
| 02 | [安装指南](./02-安装指南.md) | 环境要求、安装步骤、验证方式与常见问题 |
| 03 | [架构设计](./03-架构设计.md) | 分层架构图、模块依赖、训练数据流与计算图 |
| 04 | [NdArray 模块](./04-NdArray模块.md) | 多维数组、形状管理、广播与数值运算 |
| 05 | [Autograd 模块](./05-Autograd模块.md) | `Tensor`、`Function`、动态计算图与反向传播 |
| 06 | [神经网络 (nn) 模块](./06-神经网络nn模块.md) | `Module`、`Parameter`、容器与各类网络层 |
| 07 | [训练框架 (ml) 模块](./07-训练框架ml模块.md) | `Model`、`Trainer`、优化器、损失函数与评估器 |
| 08 | [工具与数据加载 (utils) 模块](./08-工具与数据加载utils模块.md) | `Dataset`、`DataLoader`、`Sampler`、随机数工具 |
| 09 | [API 参考文档](./09-API参考文档.md) | 核心公共 API 的签名、参数与返回值速查 |
| 10 | [常见问题解答 (FAQ)](./10-常见问题解答FAQ.md) | 使用过程中高频问题的排查与解决方案 |
| 11 | [贡献指南](./11-贡献指南.md) | 开发流程、代码规范、测试要求与 PR 流程 |
| 12 | [强化学习 (rl) 模块](./12-强化学习rl模块.md) | 环境、空间、回放、轨迹缓存与主流深度强化学习算法 |

## 🚀 快速入门三步走

1. **安装** → 阅读 [安装指南](./02-安装指南.md)
2. **跑通第一个示例** → 阅读 [项目概述](./01-项目概述.md) 中的"快速上手"部分
3. **深入理解原理** → 阅读 [架构设计](./03-架构设计.md) + [Autograd 模块](./05-Autograd模块.md)

## 🗺️ 推荐阅读路径

### 学习者路径（理解框架原理）

```
01 项目概述  →  03 架构设计  →  04 NdArray  →  05 Autograd  →  06 nn  →  07 ml
```

### 使用者路径（只想跑模型）

```
02 安装指南  →  01 项目概述（快速上手）  →  09 API 参考  →  10 FAQ
```

### 贡献者路径（想参与开发）

```
03 架构设计  →  06/07 对应模块  →  11 贡献指南
```

### 强化学习路径（环境交互与策略优化）

```
01 项目概述  →  06 nn  →  07 ml  →  12 rl  →  examples/rl
```

## 🧭 项目结构速览

```text
tinyTorch/
├── tinytorch/           # 框架主源码
│   ├── ndarr/           # 多维数组层（NdArray / Shape）
│   ├── autograd/        # 自动微分引擎（Tensor / Function / ops / graph_viz）
│   ├── nn/              # 神经网络层（module / parameter / container / init / layers）
│   ├── ml/              # 训练框架（model / trainer / dataset /
│   │                    #           optimizers / losses / evaluators /
│   │                    #           monitor / visualizer）
│   ├── rl/              # 强化学习（spaces / envs / replay / agents）
│   ├── utils/           # DataLoader、Sampler、随机数工具
│   └── constants.py     # 数值稳定性常量（epsilon、exp 溢出阈值等）
├── examples/            # 可运行示例（basic/cnn/nlp/transformer/deepseek/visualization/rl）
├── docs/                # 教程与技术方案
│   ├── tutorials/       #   入门教程（quickstart / tensor / autograd）
│   └── 方案.md          #   技术方案文档
├── tests/               # pytest 单元测试
├── run_tests.py         # 测试入口脚本（自动安装 pytest 并转发参数）
├── pyproject.toml       # 构建与工具配置（ruff / pytest / coverage）
└── wiki/                # 本 Wiki 文档
```

## 📮 反馈与支持

- 发现文档错误？欢迎按 [贡献指南](./11-贡献指南.md) 提交修复。
- 使用中遇到问题？先查阅 [FAQ](./10-常见问题解答FAQ.md)，必要时提交 Issue。

---

**版本**：`0.1.0` （Alpha）  ·  **Python**：`>=3.7`  ·  **License**：MIT
