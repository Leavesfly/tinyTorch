"""Autograd 运算的共享工具函数。

用于处理广播、梯度归约等反向传播中的通用逻辑。
"""

from tinytorch.ndarr import NdArray


def sum_to_shape(tensor: NdArray, target_shape) -> NdArray:
    """将 ndarr 求和到 target_shape（处理反向传播中的广播）。

    Args:
        tensor: 梯度张量
        target_shape: 目标形状（Shape 或兼容对象）

    Returns:
        归约后的梯度，形状与 target_shape 一致
    """
    # 对额外维度求和
    ndim_diff = tensor.shape.ndim - target_shape.ndim
    for _ in range(ndim_diff):
        tensor = tensor.sum(axis=0, keepdims=False)

    # 对目标为 1 的维度求和
    for i in range(target_shape.ndim):
        if target_shape[i] == 1 and tensor.shape[i] > 1:
            tensor = tensor.sum(axis=i, keepdims=True)

    return tensor
