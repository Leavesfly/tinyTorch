"""强化学习经验回放组件。"""

from collections import deque
from typing import Deque, Iterable, List, NamedTuple, Sequence

from tinytorch.utils import random as tt_random


class Transition(NamedTuple):
    """单步环境交互样本。"""

    state: Sequence[float]
    action: int
    reward: float
    next_state: Sequence[float]
    done: bool


class ReplayBuffer:
    """固定容量经验回放缓冲区。"""

    def __init__(self, capacity: int = 10000):
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive")
        self.capacity = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, state: Sequence[float], action: int, reward: float, next_state: Sequence[float], done: bool) -> None:
        """写入一条 transition。"""
        self._buffer.append(Transition(state, action, float(reward), next_state, bool(done)))

    def extend(self, transitions: Iterable[Transition]) -> None:
        """批量写入 transition。"""
        for transition in transitions:
            self._buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        """无放回随机采样一个 batch。"""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._buffer):
            raise ValueError("Cannot sample more transitions than the buffer contains")

        indices = list(range(len(self._buffer)))
        tt_random.shuffle(indices)
        return [self._buffer[i] for i in indices[:batch_size]]

    def clear(self) -> None:
        """清空缓冲区。"""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self)}, capacity={self.capacity})"
