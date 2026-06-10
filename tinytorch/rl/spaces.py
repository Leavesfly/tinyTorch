"""强化学习空间定义。

当前仅提供离散动作空间，覆盖教学示例中最常见的 tabular RL 和 DQN 场景。
"""

from tinytorch.utils import random as tt_random


class Discrete:
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
