"""强化学习空间定义。

提供简化版 Gym 风格空间，覆盖离散动作、连续向量观测和多离散动作。
"""

from typing import List, Sequence

from tinytorch.utils import random as tt_random


class Space:
    """空间基类。"""

    def sample(self):
        """随机采样一个合法值。"""
        raise NotImplementedError

    def contains(self, value) -> bool:
        """检查值是否属于该空间。"""
        raise NotImplementedError


class Discrete(Space):
    """离散空间，取值范围为 ``0`` 到 ``n - 1``。"""

    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("Discrete space size must be positive")
        self.n = n

    def sample(self) -> int:
        """随机采样一个合法值。"""
        return int(tt_random.random() * self.n)

    def contains(self, value: int) -> bool:
        """检查值是否属于该空间。"""
        return isinstance(value, int) and 0 <= value < self.n

    def __repr__(self) -> str:
        return f"Discrete(n={self.n})"


class Box(Space):
    """连续向量空间。

    Args:
        low: 每个维度的下界，传入标量时会广播到所有维度
        high: 每个维度的上界，传入标量时会广播到所有维度
        shape: 空间形状，当前支持一维向量
    """

    def __init__(self, low, high, shape):
        if len(shape) != 1:
            raise NotImplementedError("Box currently supports 1D vector spaces only")
        if shape[0] <= 0:
            raise ValueError("Box shape dimensions must be positive")

        self.shape = tuple(shape)
        self.low = self._expand_bound(low, self.shape[0], "low")
        self.high = self._expand_bound(high, self.shape[0], "high")
        for lo, hi in zip(self.low, self.high):
            if lo > hi:
                raise ValueError("Box low values must be <= high values")

    @staticmethod
    def _expand_bound(bound, size: int, name: str) -> List[float]:
        if isinstance(bound, (int, float)):
            return [float(bound) for _ in range(size)]
        values = list(bound)
        if len(values) != size:
            raise ValueError(f"Box {name} length must match shape")
        return [float(value) for value in values]

    def sample(self) -> List[float]:
        """均匀采样一个连续向量。"""
        return [tt_random.uniform(lo, hi) for lo, hi in zip(self.low, self.high)]

    def contains(self, value) -> bool:
        """检查向量是否落在边界内。"""
        if not isinstance(value, (list, tuple)) or len(value) != self.shape[0]:
            return False
        for item, lo, hi in zip(value, self.low, self.high):
            if not isinstance(item, (int, float)) or item < lo or item > hi:
                return False
        return True

    def __repr__(self) -> str:
        return f"Box(low={self.low}, high={self.high}, shape={self.shape})"


class MultiDiscrete(Space):
    """多离散空间，每个维度都有独立的离散取值数量。"""

    def __init__(self, nvec: Sequence[int]):
        self.nvec = [int(n) for n in nvec]
        if not self.nvec or any(n <= 0 for n in self.nvec):
            raise ValueError("MultiDiscrete nvec must contain positive integers")

    def sample(self) -> List[int]:
        """随机采样一个多离散动作。"""
        return [int(tt_random.random() * n) for n in self.nvec]

    def contains(self, value) -> bool:
        """检查值是否属于该空间。"""
        if not isinstance(value, (list, tuple)) or len(value) != len(self.nvec):
            return False
        return all(isinstance(item, int) and 0 <= item < n for item, n in zip(value, self.nvec))

    def __repr__(self) -> str:
        return f"MultiDiscrete(nvec={self.nvec})"
