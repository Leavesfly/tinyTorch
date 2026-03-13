"""Autograd loss operations.

This module contains differentiable loss primitives that are reused by the
high-level loss wrappers in ``tinytorch.ml.losses``.
"""

import math
from typing import List

from tinytorch.autograd.function import Function
from tinytorch.ndarr import NdArray
from tinytorch.ndarr.shape import Shape


class CrossEntropy(Function):
    """Cross entropy over logits and class indices."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: NdArray, target: NdArray) -> NdArray:
        if len(logits.shape.dims) != 2:
            raise ValueError(f"Expected 2D logits, got shape {logits.shape.dims}")
        if len(target.shape.dims) != 1:
            raise ValueError(f"Expected 1D target, got shape {target.shape.dims}")

        batch_size, num_classes = logits.shape.dims
        if target.shape.dims[0] != batch_size:
            raise ValueError(
                "Target batch size must match logits batch size, "
                f"got {target.shape.dims[0]} and {batch_size}"
            )

        probs = [0.0] * logits.shape.size
        losses = [0.0] * batch_size
        self._target_classes = []

        for b in range(batch_size):
            row_start = b * num_classes
            row_end = row_start + num_classes
            row = logits.data[row_start:row_end]
            max_logit = max(row)
            exp_row = [math.exp(value - max_logit) for value in row]
            denom = sum(exp_row)

            for c, exp_value in enumerate(exp_row):
                probs[row_start + c] = exp_value / denom

            target_class = int(target.data[b])
            if not 0 <= target_class < num_classes:
                raise IndexError(
                    f"Target class {target_class} out of range [0, {num_classes})"
                )
            self._target_classes.append(target_class)

            prob = max(probs[row_start + target_class], 1e-10)
            losses[b] = -math.log(prob)

        self._batch_size = batch_size
        self._num_classes = num_classes
        self._probs = probs
        self.save_for_backward(logits, target)

        if self.reduction == "none":
            return NdArray(losses, Shape((batch_size,)), logits.dtype)

        total_loss = sum(losses)
        if self.reduction == "mean":
            total_loss /= batch_size

        return NdArray([total_loss], Shape((1,)), logits.dtype)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        logits, target = self.get_saved_tensors()
        grad_logits = self._probs.copy()

        for b, target_class in enumerate(self._target_classes):
            grad_logits[b * self._num_classes + target_class] -= 1.0

        if self.reduction == "none":
            if grad_output.shape.dims != (self._batch_size,):
                raise ValueError(
                    "grad_output for reduction='none' must have shape "
                    f"({self._batch_size},), got {grad_output.shape.dims}"
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
    """Binary cross entropy over probabilities."""

    def __init__(self, reduction: str = "mean", epsilon: float = 1e-10):
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, input: NdArray, target: NdArray) -> NdArray:
        if input.shape != target.shape:
            raise ValueError(
                "input and target must have the same shape, "
                f"got {input.shape.dims} and {target.shape.dims}"
            )

        clipped = []
        losses = []
        for pred, tgt in zip(input.data, target.data):
            clipped_pred = max(self.epsilon, min(1.0 - self.epsilon, pred))
            clipped.append(clipped_pred)
            losses.append(
                -(tgt * math.log(clipped_pred) + (1.0 - tgt) * math.log(1.0 - clipped_pred))
            )

        self._clipped_input = clipped
        self._element_count = input.shape.size
        self.save_for_backward(input, target)

        if self.reduction == "none":
            return NdArray(losses, input.shape, input.dtype)

        total_loss = sum(losses)
        if self.reduction == "mean":
            total_loss /= self._element_count

        return NdArray([total_loss], Shape((1,)), input.dtype)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        input, target = self.get_saved_tensors()
        grad_input = [0.0] * self._element_count

        if self.reduction == "none":
            if grad_output.shape != input.shape:
                raise ValueError(
                    "grad_output for reduction='none' must match input shape, "
                    f"got {grad_output.shape.dims} and {input.shape.dims}"
                )
            upstream = grad_output.data
            mean_scale = 1.0
        else:
            upstream = [grad_output.data[0]] * self._element_count
            mean_scale = 1.0 / self._element_count if self.reduction == "mean" else 1.0

        for i, (pred, tgt) in enumerate(zip(self._clipped_input, target.data)):
            grad = (pred - tgt) / (pred * (1.0 - pred))
            grad_input[i] = grad * upstream[i] * mean_scale

        grad_target = NdArray.zeros(target.shape, target.dtype)
        return [
            NdArray(grad_input, input.shape, input.dtype),
            grad_target,
        ]
