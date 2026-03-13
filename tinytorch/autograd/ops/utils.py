"""自动微分运算的共享工具函数。

提供反向传播中的通用逻辑，包括:
    - sum_to_shape: 梯度归约至目标形状 (处理广播)
"""

from tinytorch.ndarr import NdArray


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
    # 例如: (1, 3, 4) -> (3, 4)，在 axis=0 上求和
    ndim_diff = tensor.shape.ndim - target_shape.ndim
    for _ in range(ndim_diff):
        tensor = tensor.sum(axis=0, keepdims=False)

    # Step 2: 对目标形状中大小为 1 的维度求和
    # 例如: (3, 4) -> (3, 1)，在 axis=1 上求和并保持维度
    for i in range(target_shape.ndim):
        if target_shape[i] == 1 and tensor.shape[i] > 1:
            tensor = tensor.sum(axis=i, keepdims=True)

    return tensor
