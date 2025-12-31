# tinyTorch 教程

欢迎来到 tinyTorch 学习之旅！这些教程将带你从零开始，深入理解深度学习框架的内部实现。

## 📚 教程目录

### 入门篇

1. **[快速入门](01_quickstart.md)**
   - tinyTorch 简介
   - 安装和环境配置
   - 核心概念：Tensor、Variable、Module
   - 第一个完整示例：线性回归
   - 预计学习时间：30 分钟

2. **[Tensor 操作详解](02_tensor_operations.md)**
   - 张量的创建和基本运算
   - 形状操作：reshape、squeeze、flatten
   - 索引和切片
   - 广播机制
   - 数学函数和激活函数
   - 预计学习时间：1 小时

3. **[自动微分系统](03_autograd_system.md)**
   - 什么是自动微分
   - Variable 和 Function
   - 计算图和反向传播
   - 常用操作的梯度
   - 梯度检查和调试
   - 预计学习时间：1.5 小时

### 进阶篇

4. **[构建神经网络](04_building_networks.md)**
   - Module 系统详解
   - 内置层：Linear、Conv2d、RNN、LSTM
   - 激活函数和归一化层
   - Sequential 容器
   - 自定义层
   - 预计学习时间：1 小时

5. **[训练深度学习模型](05_training_models.md)**
   - 训练循环详解
   - 优化器：SGD、Adam
   - 损失函数：MSE、CrossEntropy
   - 数据加载和批处理
   - 模型保存和加载
   - 预计学习时间：1.5 小时

### 实战篇

6. **[卷积神经网络](06_convolutional_networks.md)**
   - CNN 架构详解
   - Conv2d 层的实现原理
   - 经典 CNN 模型：LeNet、AlexNet
   - 图像分类实战
   - 预计学习时间：2 小时

7. **[循环神经网络](07_recurrent_networks.md)**
   - RNN 基础
   - LSTM 和 GRU
   - 序列建模
   - 语言模型实战
   - 预计学习时间：2 小时

8. **[Transformer 和注意力机制](08_attention_transformer.md)**
   - 注意力机制原理
   - MultiHeadAttention 实现
   - Transformer 架构
   - 文本生成实战
   - 预计学习时间：2.5 小时

## 🎯 学习路径

### 路径 1：快速上手（适合有深度学习基础）
```
01_quickstart.md → 04_building_networks.md → 05_training_models.md
```
总时间：~3 小时

### 路径 2：深入理解（适合想了解底层原理）
```
01_quickstart.md → 02_tensor_operations.md → 03_autograd_system.md 
→ 04_building_networks.md → 05_training_models.md
```
总时间：~5.5 小时

### 路径 3：完整学习（适合从零开始）
按顺序学完所有教程
总时间：~12 小时

### 路径 4：专题学习
- **计算机视觉**：01 → 04 → 05 → 06
- **自然语言处理**：01 → 04 → 05 → 07 → 08
- **框架开发**：02 → 03 → 自定义实现

## 🛠️ 实践项目

每个教程都包含：
- ✅ **理论讲解**：清晰的概念说明
- 💻 **代码示例**：可运行的完整代码
- 🎓 **练习题**：巩固所学知识
- 🔍 **调试技巧**：常见问题和解决方法

## 📖 配套资源

- **示例代码**：[../examples/](../examples/)
- **API 文档**：[../docs/api_reference.md](../docs/api_reference.md)
- **架构设计**：[../docs/architecture.md](../docs/architecture.md)
- **技术方案**：[../docs/方案.md](../docs/方案.md)

## 💡 学习建议

1. **动手实践**：每个示例代码都要自己运行一遍
2. **完成练习**：练习题能帮助你巩固知识
3. **理解原理**：不要只记 API，要理解背后的数学和算法
4. **对比学习**：可以对照 PyTorch 理解概念
5. **循序渐进**：不要跳过基础教程

## 🤔 常见问题

**Q: 我需要什么基础知识？**  
A: 
- Python 编程基础（必需）
- 线性代数基础（推荐）
- 微积分基础（推荐）
- 深度学习概念（可选，教程会讲解）

**Q: tinyTorch 和 PyTorch 的区别？**  
A: tinyTorch 是纯 Python 实现，不依赖任何库，专注教学。PyTorch 是高性能的生产级框架。

**Q: 学完后我能做什么？**  
A: 
- 深入理解深度学习框架的工作原理
- 能够阅读和理解 PyTorch/TensorFlow 源码
- 实现自定义的深度学习操作
- 为开源深度学习框架做贡献

**Q: 代码运行很慢怎么办？**  
A: 这是正常的，纯 Python 实现本身就慢。tinyTorch 是为了学习，不是为了性能。

## 📝 反馈和贡献

如果你发现教程中的错误或有改进建议，欢迎：
- 提交 Issue
- 提交 Pull Request
- 分享你的学习心得

## 🚀 开始学习

准备好了吗？从 [01_quickstart.md](01_quickstart.md) 开始你的 tinyTorch 之旅吧！

---

**祝学习愉快！**  
TinyAI Team
