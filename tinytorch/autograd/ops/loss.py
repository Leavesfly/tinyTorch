"""损失函数运算 - 支持自动微分。

本模块提供可微分的损失函数原语，供高层 tinytorch.ml.losses 模块使用。

包含以下损失函数:
    - CrossEntropy:       交叉熵损失 (用于多分类)
    - BinaryCrossEntropy: 二元交叉熵损失 (用于二分类)
"""

import math
from typing import List

from tinytorch.autograd.function import Function
from tinytorch.ndarr import NdArray
from tinytorch.ndarr.shape import Shape


class CrossEntropy(Function):
    """交叉熵损失函数。

    计算预测 logits 与目标类别之间的交叉熵损失。

    数学表达式:
        loss = -log(softmax(logits)[target_class])

    参数:
        reduction: 归约方式
            - "mean": 返回批次平均损失
            - "sum": 返回批次总损失
            - "none": 返回每个样本的损失

    形状说明:
        - logits: (batch_size, num_classes) 未归一化的预测分数
        - target: (batch_size,) 目标类别索引
        - output: 标量 (mean/sum) 或 (batch_size,) (none)

    注意:
        - 内部使用 Softmax 数值稳定实现，避免溢出
        - 梯度公式: grad_logits = softmax(logits) - one_hot(target)
    """

    def __init__(self, reduction: str = "mean"):
        """初始化交叉熵损失。

        Args:
            reduction: 归约方式，可选 "mean"、"sum" 或 "none"
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: NdArray, target: NdArray) -> NdArray:
        """前向传播: 计算交叉熵损失。"""
        if len(logits.shape.dims) != 2:
            raise ValueError(f"logits 应为 2D 张量，得到形状 {logits.shape.dims}")
        if len(target.shape.dims) != 1:
            raise ValueError(f"target 应为 1D 张量，得到形状 {target.shape.dims}")

        batch_size, num_classes = logits.shape.dims
        if target.shape.dims[0] != batch_size:
            raise ValueError(
                f"target 批次大小 {target.shape.dims[0]} 与 logits 批次大小 {batch_size} 不匹配"
            )

        probs = [0.0] * logits.shape.size
        losses = [0.0] * batch_size
        self._target_classes = []

        for b in range(batch_size):
            row_start = b * num_classes
            row_end = row_start + num_classes
            row = logits.data[row_start:row_end]

            # Softmax (数值稳定实现)
            max_logit = max(row)
            exp_row = [math.exp(value - max_logit) for value in row]
            denom = sum(exp_row)

            for c, exp_value in enumerate(exp_row):
                probs[row_start + c] = exp_value / denom

            # 获取目标类别并验证
            target_class = int(target.data[b])
            if not 0 <= target_class < num_classes:
                raise IndexError(f"目标类别 {target_class} 超出范围 [0, {num_classes})")
            self._target_classes.append(target_class)

            # 计算交叉熵损失
            prob = max(probs[row_start + target_class], 1e-10)
            losses[b] = -math.log(prob)

        self._batch_size = batch_size
        self._num_classes = num_classes
        self._probs = probs
        self.save_for_backward(logits, target)

        # 应用归约
        if self.reduction == "none":
            return NdArray(losses, Shape((batch_size,)), logits.dtype)

        total_loss = sum(losses)
        if self.reduction == "mean":
            total_loss /= batch_size

        return NdArray([total_loss], Shape((1,)), logits.dtype)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算 logits 梯度。

        梯度公式: grad_logits = softmax(logits) - one_hot(target)
        """
        logits, target = self.get_saved_tensors()
        grad_logits = self._probs.copy()

        # 在目标类别位置减 1 (softmax - one_hot)
        for b, target_class in enumerate(self._target_classes):
            grad_logits[b * self._num_classes + target_class] -= 1.0

        # 应用归约和上游梯度
        if self.reduction == "none":
            if grad_output.shape.dims != (self._batch_size,):
                raise ValueError(
                    f"reduction='none' 时 grad_output 形状应为 ({self._batch_size},)，"
                    f"得到 {grad_output.shape.dims}"
                )
            for b in range(self._batch_size):
                scale = grad_output.data[b]
                row_start = b * self._num_classes
                for c in range(self._num_classes):
                    grad_logits[row_start + c] *= scale
        else:
            scale = grad_output.data[0]
            if self.reduction == "mean":
                scale /= self._batch_size
            grad_logits = [value * scale for value in grad_logits]

        grad_target = NdArray.zeros(target.shape, target.dtype)
        return [
            NdArray(grad_logits, logits.shape, logits.dtype),
            grad_target,
        ]


class BinaryCrossEntropy(Function):
    """二元交叉熵损失函数。

    计算预测概率与目标标签之间的二元交叉熵损失。

    数学表达式:
        loss = -[y * log(p) + (1-y) * log(1-p)]

    参数:
        reduction: 归约方式
            - "mean": 返回元素平均损失
            - "sum": 返回元素总损失
            - "none": 返回每个元素的损失
        epsilon: 数值稳定性常数，裁剪预测值范围

    形状说明:
        - input:  任意形状，表示预测概率
        - target:  与 input 相同形状，表示目标标签 (0 或 1)
        - output:  标量 (mean/sum) 或与 input 相同形状 (none)

    注意:
        - 输入会被裁剪到 [epsilon, 1-epsilon] 避免数值问题
        - 梯度公式: grad_input = (p - y) / (p * (1-p))
    """

    def __init__(self, reduction: str = "mean", epsilon: float = 1e-10):
        """初始化二元交叉熵损失。

        Args:
            reduction: 归约方式，可选 "mean"、"sum" 或 "none"
            epsilon: 数值稳定性常数
        """
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, input: NdArray, target: NdArray) -> NdArray:
        """前向传播: 计算二元交叉熵损失。"""
        if input.shape != target.shape:
            raise ValueError(
                f"input 和 target 形状必须相同，得到 {input.shape.dims} 和 {target.shape.dims}"
            )

        clipped = []
        losses = []
        for pred, tgt in zip(input.data, target.data):
            # 裁剪预测值以避免 log(0)
            clipped_pred = max(self.epsilon, min(1.0 - self.epsilon, pred))
            clipped.append(clipped_pred)
            # 计算二元交叉熵
            losses.append(
                -(tgt * math.log(clipped_pred) + (1.0 - tgt) * math.log(1.0 - clipped_pred))
            )

        self._clipped_input = clipped
        self._element_count = input.shape.size
        self.save_for_backward(input, target)

        # 应用归约
        if self.reduction == "none":
            return NdArray(losses, input.shape, input.dtype)

        total_loss = sum(losses)
        if self.reduction == "mean":
            total_loss /= self._element_count

        return NdArray([total_loss], Shape((1,)), input.dtype)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算 input 梯度。

        梯度公式: grad_input = (p - y) / (p * (1-p))
        """
        input, target = self.get_saved_tensors()
        grad_input = [0.0] * self._element_count

        # 处理归约和上游梯度
        if self.reduction == "none":
            if grad_output.shape != input.shape:
                raise ValueError(
                    f"reduction='none' 时 grad_output 形状应与 input 相同，"
                    f"得到 {grad_output.shape.dims} 和 {input.shape.dims}"
                )
            upstream = grad_output.data
            mean_scale = 1.0
        else:
            upstream = [grad_output.data[0]] * self._element_count
            mean_scale = 1.0 / self._element_count if self.reduction == "mean" else 1.0

        # 计算梯度
        for i, (pred, tgt) in enumerate(zip(self._clipped_input, target.data)):
            grad = (pred - tgt) / (pred * (1.0 - pred))
            grad_input[i] = grad * upstream[i] * mean_scale

        grad_target = NdArray.zeros(target.shape, target.dtype)
        return [
            NdArray(grad_input, input.shape, input.dtype),
            grad_target,
        ]
