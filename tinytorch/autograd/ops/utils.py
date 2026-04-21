"""自动微分运算的共享工具函数。

提供反向传播中的通用逻辑，包括:
    - sum_to_shape: 梯度归约至目标形状 (处理广播)
    - broadcast_backward: 二元运算的广播梯度归约
    - expand_and_broadcast: 归约运算的梯度广播回原始形状
    - DEFAULT_EPSILON: 数值稳定性默认常量
"""

from typing import Optional
from tinytorch.ndarr import NdArray

DEFAULT_EPSILON: float = 1e-10

def sum_to_shape(tensor: NdArray, target_shape) -> NdArray:
    """将张量求和归约至目标形状。

    在反向传播中处理广播机制: 当前向传播发生广播时，
    反向传播需要将梯度在广播维度上求和。

    Args:
        tensor: 待归约的梯度张量
        target_shape: 目标形状 (Shape 对象或兼容类型)

    Returns:
        归约后的梯度张量，形状与 target_shape 一致

    示例:
        >>> # 前向传播: (3, 4) + (1, 4) -> (3, 4)
        >>> # 反向传播: grad (3, 4) -> sum_to_shape -> (1, 4)
        >>> grad = NdArray([...], Shape((3, 4)))
        >>> sum_to_shape(grad, Shape((1, 4)))  # 在 axis=0 上求和
    """
    # Step 1: 消除额外的前置维度
    ndim_diff = tensor.shape.ndim - target_shape.ndim
    for _ in range(ndim_diff):
        tensor = tensor.sum(axis=0, keepdims=False)

    # Step 2: 对目标形状中大小为 1 的维度求和
    for i in range(target_shape.ndim):
        if target_shape[i] == 1 and tensor.shape[i] > 1:
            tensor = tensor.sum(axis=i, keepdims=True)

    return tensor

def broadcast_backward(grad: NdArray, target_shape) -> NdArray:
    """二元运算反向传播的广播梯度归约。

    当前向传播中发生了广播时，反向传播需要将梯度归约回原始输入形状。
    如果梯度形状已与目标形状一致，则直接返回。

    Args:
        grad: 待归约的梯度
        target_shape: 目标输入的原始形状

    Returns:
        归约后的梯度，形状与 target_shape 一致
    """
    if grad.shape == target_shape:
        return grad
    return sum_to_shape(grad, target_shape)

def expand_and_broadcast(grad_output: NdArray, input_shape, axis: Optional[int], keepdims: bool) -> NdArray:
    """归约运算反向传播的梯度广播。

    将归约运算产生的梯度广播回原始输入形状。根据 keepdims 和 axis
    参数采取不同的扩展策略。

    Args:
        grad_output: 归约运算输出的梯度
        input_shape: 前向传播时输入的原始形状
        axis: 归约轴，None 表示对所有维度归约
        keepdims: 前向传播时是否保留了归约维度

    Returns:
        广播回输入形状的梯度
    """
    if keepdims:
        return grad_output._broadcast_to(input_shape)

    if axis is None:
        return NdArray(
            [grad_output.data[0]] * input_shape.size,
            input_shape, grad_output.dtype
        )

    normalized_axis = axis if axis >= 0 else input_shape.ndim + axis
    expanded_dims = list(grad_output.shape.dims)
    expanded_dims.insert(normalized_axis, 1)
    grad_expanded = grad_output.reshape(tuple(expanded_dims))
    return grad_expanded._broadcast_to(input_shape)