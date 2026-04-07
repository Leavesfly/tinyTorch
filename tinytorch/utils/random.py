"""Shared random-number utilities for tinyTorch.

The module centralizes stochastic behavior so initialization, dropout, and data
shuffling follow the same RNG source by default.

线程安全说明：
- 本模块使用全局随机数生成器 _GLOBAL_RNG，在多线程环境下不是线程安全的
- 如果需要在多线程环境中使用，请为每个线程创建独立的 RNG 实例（使用 generator() 函数）
- seed() 函数会影响全局状态，在多线程环境下调用可能导致不可预期的结果
"""

import random as _py_random
from typing import MutableSequence, Optional, Any


_GLOBAL_RNG = _py_random.Random()


def seed(value: int) -> None:
    """Seed the shared tinyTorch RNG.
    
    Args:
        value: 随机数种子值
    """
    _GLOBAL_RNG.seed(value)


def generator(seed_value: Optional[int] = None) -> _py_random.Random:
    """Create an independent RNG, optionally seeded.
    
    Args:
        seed_value: 可选的随机数种子值，如果为 None 则使用系统随机源
        
    Returns:
        独立的随机数生成器实例
    """
    rng = _py_random.Random()
    if seed_value is not None:
        rng.seed(seed_value)
    return rng


def random() -> float:
    """Draw a uniform random float in ``[0, 1)``.
    
    Returns:
        [0, 1) 范围内的随机浮点数
    """
    return _GLOBAL_RNG.random()


def uniform(a: float, b: float) -> float:
    """Draw a uniform random value from ``[a, b]``.
    
    Args:
        a: 下界（包含）
        b: 上界（包含）
        
    Returns:
        [a, b] 范围内的均匀分布随机数
    """
    return _GLOBAL_RNG.uniform(a, b)


def gauss(mean: float, std: float) -> float:
    """Draw a Gaussian random value.
    
    Args:
        mean: 均值
        std: 标准差
        
    Returns:
        服从高斯分布的随机数
    """
    return _GLOBAL_RNG.gauss(mean, std)


def shuffle(values: MutableSequence[Any]) -> None:
    """Shuffle a mutable sequence in place.
    
    Args:
        values: 可变序列，将在原地进行打乱
    """
    _GLOBAL_RNG.shuffle(values)
