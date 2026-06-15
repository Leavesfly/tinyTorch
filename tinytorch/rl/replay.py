"""强化学习经验回放与轨迹缓存组件。"""

from collections import deque
from typing import Deque, Iterable, List, NamedTuple, Optional, Sequence

from tinytorch.utils import random as tt_random


class Transition(NamedTuple):
    """单步环境交互样本。"""

    state: Sequence[float]
    action: int
    reward: float
    next_state: Sequence[float]
    done: bool


class PrioritizedSample(NamedTuple):
    """优先级回放采样结果。"""

    transitions: List[Transition]
    indices: List[int]
    weights: List[float]


class RolloutStep(NamedTuple):
    """on-policy 轨迹中的一步样本。"""

    state: Sequence[float]
    action: int
    reward: float
    next_state: Sequence[float]
    done: bool
    value: float
    log_prob: float


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


class PrioritizedReplayBuffer(ReplayBuffer):
    """按 TD error 优先采样的经验回放缓冲区。

    这是纯 Python 教学版 PER，使用简单的线性扫描采样，适合小规模实验。
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4, epsilon: float = 1e-6):
        super().__init__(capacity=capacity)
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self._priorities: Deque[float] = deque(maxlen=capacity)

    def push(
        self,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Sequence[float],
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        """写入一条 transition，并设置初始优先级。"""
        super().push(state, action, reward, next_state, done)
        if priority is None:
            priority = max(self._priorities) if self._priorities else 1.0
        self._priorities.append(float(priority))

    def extend(self, transitions: Iterable[Transition]) -> None:
        """批量写入 transition。"""
        for transition in transitions:
            self.push(transition.state, transition.action, transition.reward, transition.next_state, transition.done)

    def sample(self, batch_size: int) -> PrioritizedSample:
        """按优先级采样一个 batch。"""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._buffer):
            raise ValueError("Cannot sample more transitions than the buffer contains")

        scaled = [(priority + self.epsilon) ** self.alpha for priority in self._priorities]
        total = sum(scaled)
        probs = [value / total for value in scaled]

        indices = []
        for _ in range(batch_size):
            indices.append(self._sample_index(probs))

        min_prob = min(probs)
        max_weight = (len(self._buffer) * min_prob) ** (-self.beta) if min_prob > 0 else 1.0
        weights = []
        for idx in indices:
            weight = (len(self._buffer) * probs[idx]) ** (-self.beta) if probs[idx] > 0 else 1.0
            weights.append(weight / max_weight)

        return PrioritizedSample(
            transitions=[self._buffer[idx] for idx in indices],
            indices=indices,
            weights=weights,
        )

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float]) -> None:
        """根据最新 TD error 更新样本优先级。"""
        if len(indices) != len(priorities):
            raise ValueError("indices and priorities must have the same length")
        priority_list = list(self._priorities)
        for idx, priority in zip(indices, priorities):
            if not 0 <= idx < len(priority_list):
                raise IndexError("priority index out of range")
            priority_list[idx] = abs(float(priority)) + self.epsilon
        self._priorities = deque(priority_list, maxlen=self.capacity)

    def clear(self) -> None:
        """清空缓冲区和优先级。"""
        super().clear()
        self._priorities.clear()

    @staticmethod
    def _sample_index(probs: Sequence[float]) -> int:
        threshold = tt_random.random()
        cumulative = 0.0
        for idx, prob in enumerate(probs):
            cumulative += prob
            if threshold <= cumulative:
                return idx
        return len(probs) - 1


class RolloutBuffer:
    """on-policy 轨迹缓存，支持 discounted return 和 GAE。"""

    def __init__(self):
        self.steps: List[RolloutStep] = []
        self.returns: List[float] = []
        self.advantages: List[float] = []

    def add(
        self,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Sequence[float],
        done: bool,
        value: float = 0.0,
        log_prob: float = 0.0,
    ) -> None:
        """追加一步 on-policy 样本。"""
        self.steps.append(RolloutStep(state, action, float(reward), next_state, bool(done), float(value), float(log_prob)))

    def compute_returns_and_advantages(
        self,
        last_value: float = 0.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = True,
    ) -> None:
        """计算 GAE advantage 和 return。"""
        self.advantages = [0.0 for _ in self.steps]
        self.returns = [0.0 for _ in self.steps]
        gae = 0.0

        for idx in range(len(self.steps) - 1, -1, -1):
            step = self.steps[idx]
            next_value = last_value if idx == len(self.steps) - 1 else self.steps[idx + 1].value
            non_terminal = 0.0 if step.done else 1.0
            delta = step.reward + gamma * next_value * non_terminal - step.value
            gae = delta + gamma * gae_lambda * non_terminal * gae
            self.advantages[idx] = gae
            self.returns[idx] = gae + step.value

        if normalize_advantages and self.advantages:
            mean = sum(self.advantages) / len(self.advantages)
            variance = sum((adv - mean) ** 2 for adv in self.advantages) / len(self.advantages)
            std = variance ** 0.5
            if std > 1e-8:
                self.advantages = [(adv - mean) / (std + 1e-8) for adv in self.advantages]

    def to_transitions(self) -> List[Transition]:
        """转换为通用 Transition 列表。"""
        return [Transition(step.state, step.action, step.reward, step.next_state, step.done) for step in self.steps]

    def action_probs(self) -> List[float]:
        """根据 log_prob 还原采样时动作概率。"""
        return [pow(2.718281828459045, step.log_prob) for step in self.steps]

    def clear(self) -> None:
        """清空缓存。"""
        self.steps = []
        self.returns = []
        self.advantages = []

    def __len__(self) -> int:
        return len(self.steps)
