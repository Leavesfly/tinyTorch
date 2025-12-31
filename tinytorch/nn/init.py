"""参数初始化方法。

提供各种常用的参数初始化策略，如 Xavier、He、正态分布、均匀分布等。

Author: TinyAI Team
"""

import math
from tinytorch.tensor.tensor import Tensor
from tinytorch.nn.parameter import Parameter


def calculate_gain(nonlinearity: str = 'linear') -> float:
    """计算激活函数的增益因子。
    
    不同的激活函数需要不同的初始化尺度。
    
    Args:
        nonlinearity: 激活函数名称，支持 'linear', 'relu', 'leaky_relu', 
                     'tanh', 'sigmoid'
    
    Returns:
        增益因子
    """
    gains = {
        'linear': 1.0,
        'relu': math.sqrt(2.0),
        'leaky_relu': math.sqrt(2.0 / (1 + 0.01**2)),
        'tanh': 5.0 / 3,
        'sigmoid': 1.0
    }
    return gains.get(nonlinearity, 1.0)


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    """使用均匀分布初始化张量（原地操作）。
    
    Args:
        tensor: 要初始化的张量
        a: 均匀分布的下界
        b: 均匀分布的上界
    
    Returns:
        初始化后的张量（同一个对象）
    """
    import random
    size = tensor.shape.size
    tensor.data = [random.uniform(a, b) for _ in range(size)]
    return tensor


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """使用正态分布初始化张量（原地操作）。
    
    Args:
        tensor: 要初始化的张量
        mean: 正态分布的均值
        std: 正态分布的标准差
    
    Returns:
        初始化后的张量（同一个对象）
    """
    import random
    size = tensor.shape.size
    tensor.data = [random.gauss(mean, std) for _ in range(size)]
    return tensor


def constant_(tensor: Tensor, val: float) -> Tensor:
    """使用常数初始化张量（原地操作）。
    
    Args:
        tensor: 要初始化的张量
        val: 常数值
    
    Returns:
        初始化后的张量（同一个对象）
    """
    size = tensor.shape.size
    tensor.data = [val] * size
    return tensor


def zeros_(tensor: Tensor) -> Tensor:
    """使用零初始化张量（原地操作）。
    
    Args:
        tensor: 要初始化的张量
    
    Returns:
        初始化后的张量（同一个对象）
    """
    return constant_(tensor, 0.0)


def ones_(tensor: Tensor) -> Tensor:
    """使用一初始化张量（原地操作）。
    
    Args:
        tensor: 要初始化的张量
    
    Returns:
        初始化后的张量（同一个对象）
    """
    return constant_(tensor, 1.0)


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """使用 Xavier 均匀分布初始化张量（原地操作）。
    
    也称为 Glorot 初始化。适用于 tanh 和 sigmoid 激活函数。
    
    公式: U(-a, a), 其中 a = gain * sqrt(6 / (fan_in + fan_out))
    
    Args:
        tensor: 要初始化的张量
        gain: 增益因子，通常使用 calculate_gain() 计算
    
    Returns:
        初始化后的张量（同一个对象）
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std  # 将标准差转换为均匀分布的边界
    return uniform_(tensor, -a, a)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """使用 Xavier 正态分布初始化张量（原地操作）。
    
    也称为 Glorot 初始化。适用于 tanh 和 sigmoid 激活函数。
    
    公式: N(0, std^2), 其中 std = gain * sqrt(2 / (fan_in + fan_out))
    
    Args:
        tensor: 要初始化的张量
        gain: 增益因子，通常使用 calculate_gain() 计算
    
    Returns:
        初始化后的张量（同一个对象）
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(tensor, 0.0, std)


def kaiming_uniform_(tensor: Tensor, a: float = 0, mode: str = 'fan_in', 
                     nonlinearity: str = 'leaky_relu') -> Tensor:
    """使用 Kaiming 均匀分布初始化张量（原地操作）。
    
    也称为 He 初始化。适用于 ReLU 激活函数。
    
    Args:
        tensor: 要初始化的张量
        a: leaky_relu 的负斜率（默认为0表示使用ReLU）
        mode: 'fan_in' 或 'fan_out'，决定使用输入还是输出神经元数量
        nonlinearity: 激活函数名称
    
    Returns:
        初始化后的张量（同一个对象）
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(tensor: Tensor, a: float = 0, mode: str = 'fan_in',
                    nonlinearity: str = 'leaky_relu') -> Tensor:
    """使用 Kaiming 正态分布初始化张量（原地操作）。
    
    也称为 He 初始化。适用于 ReLU 激活函数。
    
    Args:
        tensor: 要初始化的张量
        a: leaky_relu 的负斜率（默认为0表示使用ReLU）
        mode: 'fan_in' 或 'fan_out'，决定使用输入还是输出神经元数量
        nonlinearity: 激活函数名称
    
    Returns:
        初始化后的张量（同一个对象）
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity)
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0.0, std)


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple:
    """计算张量的 fan_in 和 fan_out。
    
    对于二维张量（权重矩阵）：
        fan_in = num_input_features
        fan_out = num_output_features
    
    对于高维张量（如卷积核）：
        fan_in = num_input_channels * kernel_size
        fan_out = num_output_channels * kernel_size
    
    Args:
        tensor: 输入张量
    
    Returns:
        (fan_in, fan_out) 元组
    """
    dimensions = len(tensor.shape.dims)
    
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    
    if dimensions == 2:  # 线性层权重 (out_features, in_features)
        fan_in = tensor.shape.dims[1]
        fan_out = tensor.shape.dims[0]
    else:  # 卷积层等 (out_channels, in_channels, kernel_h, kernel_w, ...)
        receptive_field_size = 1
        for dim in tensor.shape.dims[2:]:
            receptive_field_size *= dim
        fan_in = tensor.shape.dims[1] * receptive_field_size
        fan_out = tensor.shape.dims[0] * receptive_field_size
    
    return fan_in, fan_out


def _calculate_correct_fan(tensor: Tensor, mode: str) -> int:
    """根据模式计算正确的 fan 值。
    
    Args:
        tensor: 输入张量
        mode: 'fan_in' 或 'fan_out'
    
    Returns:
        fan 值
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        return fan_in
    elif mode == 'fan_out':
        return fan_out
    else:
        raise ValueError(f"Mode {mode} not supported, please use 'fan_in' or 'fan_out'")


# 别名，方便使用
glorot_uniform_ = xavier_uniform_
glorot_normal_ = xavier_normal_
he_uniform_ = kaiming_uniform_
he_normal_ = kaiming_normal_
