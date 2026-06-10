"""强化学习智能体实现。"""

import math
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

from tinytorch.autograd import Tensor, no_grad
from tinytorch.ml.losses import CrossEntropyLoss, MSELoss
from tinytorch.ml.optimizers import Optimizer
from tinytorch.ndarr import NdArray
from tinytorch.nn import Module
from tinytorch.rl.replay import ReplayBuffer, Transition
from tinytorch.utils import random as tt_random


StateKey = Hashable


def _argmax(values: Sequence[float]) -> int:
    best_idx = 0
    best_value = values[0]
    for idx, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_idx = idx
            best_value = value
    return best_idx


def _softmax(values: Sequence[float]) -> List[float]:
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    denom = sum(exp_values)
    return [value / denom for value in exp_values]


def _sample_from_probs(probs: Sequence[float]) -> int:
    threshold = tt_random.random()
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if threshold <= cumulative:
            return idx
    return len(probs) - 1


def _discounted_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    returns = [0.0 for _ in rewards]
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    return returns


def _normalize(values: Sequence[float], epsilon: float = 1e-8) -> List[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std < epsilon:
        return [0.0 for _ in values]
    return [(value - mean) / (std + epsilon) for value in values]


class QLearningAgent:
    """表格 Q-learning 智能体，适用于小型离散状态/动作环境。"""

    def __init__(
        self,
        action_size: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 1.0,
    ):
        if action_size <= 0:
            raise ValueError("action_size must be positive")

        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[StateKey, List[float]] = {}

    def _state_key(self, state) -> StateKey:
        if isinstance(state, list):
            return tuple(state)
        return state

    def q_values(self, state) -> List[float]:
        """获取状态对应的 Q 值列表，不存在时自动初始化为 0。"""
        key = self._state_key(state)
        if key not in self.q_table:
            self.q_table[key] = [0.0 for _ in range(self.action_size)]
        return self.q_table[key]

    def select_action(self, state, training: bool = True) -> int:
        """按 epsilon-greedy 策略选择动作。"""
        if training and tt_random.random() < self.epsilon:
            return int(tt_random.random() * self.action_size)
        return _argmax(self.q_values(state))

    def update(self, state, action: int, reward: float, next_state, done: bool) -> float:
        """执行一次 Q-learning 更新并返回 TD error。"""
        if not 0 <= action < self.action_size:
            raise ValueError(f"Invalid action {action}; expected 0 <= action < {self.action_size}")

        q_values = self.q_values(state)
        next_q_values = self.q_values(next_state)
        bootstrap = 0.0 if done else max(next_q_values)
        target = reward + self.gamma * bootstrap
        td_error = target - q_values[action]
        q_values[action] += self.learning_rate * td_error
        self._decay_epsilon()
        return td_error

    def train_episode(self, env, max_steps: Optional[int] = None) -> Tuple[float, int]:
        """在兼容 ``reset``/``step`` 的环境中训练一个 episode。"""
        state = env.reset()
        total_reward = 0.0
        steps = 0
        limit = max_steps if max_steps is not None else getattr(env, "max_steps", 1000)

        for _ in range(limit):
            action = self.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            self.update(state, action, reward, next_state, done)
            total_reward += reward
            steps += 1
            state = next_state
            if done:
                break

        return total_reward, steps

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class DQNAgent:
    """基于 tinyTorch ``Module`` 的简化 DQN 智能体。

    该实现面向教学和小规模实验：使用 MSE 让当前 Q 网络拟合 Bellman 目标，
    并通过 ``ReplayBuffer`` 打破样本相关性。
    """

    def __init__(
        self,
        q_network: Module,
        optimizer: Optimizer,
        state_dim: int,
        action_size: int,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 1.0,
        replay_buffer: Optional[ReplayBuffer] = None,
        target_network: Optional[Module] = None,
        batch_size: int = 32,
    ):
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if action_size <= 0:
            raise ValueError("action_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.q_network = q_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.state_dim = state_dim
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_buffer = replay_buffer if replay_buffer is not None else ReplayBuffer()
        self.batch_size = batch_size
        self.loss_fn = MSELoss()

        if self.target_network is not None:
            self.sync_target_network()

    def select_action(self, state: Sequence[float], training: bool = True) -> int:
        """按 epsilon-greedy 策略选择动作。"""
        if training and tt_random.random() < self.epsilon:
            return int(tt_random.random() * self.action_size)
        return _argmax(self.predict_q_values(state))

    def predict_q_values(self, state: Sequence[float]) -> List[float]:
        """返回单个状态下每个动作的 Q 值。"""
        state_values = self._as_state_list(state)
        tensor = Tensor(NdArray(state_values, (1, self.state_dim)), requires_grad=False)
        with no_grad():
            output = self.q_network(tensor)
        return list(output.value.data)

    def remember(self, state: Sequence[float], action: int, reward: float, next_state: Sequence[float], done: bool) -> None:
        """写入一条经验。"""
        self.replay_buffer.push(
            self._as_state_list(state),
            action,
            reward,
            self._as_state_list(next_state),
            done,
        )

    def learn(self, transition: Optional[Transition] = None) -> Optional[float]:
        """从经验回放中学习一次，返回 loss；样本不足时返回 ``None``。"""
        if transition is not None:
            self.replay_buffer.extend([transition])
        if len(self.replay_buffer) == 0:
            return None

        batch_size = min(self.batch_size, len(self.replay_buffer))
        batch = self.replay_buffer.sample(batch_size)
        states = self._flatten_states([item.state for item in batch])
        inputs = Tensor(NdArray(states, (batch_size, self.state_dim)), requires_grad=False)

        self.optimizer.zero_grad()
        predictions = self.q_network(inputs)
        targets = list(predictions.value.data)

        for row, item in enumerate(batch):
            if not 0 <= item.action < self.action_size:
                raise ValueError(f"Invalid action {item.action}; expected 0 <= action < {self.action_size}")
            target_value = item.reward
            if not item.done:
                target_value += self.gamma * self._bootstrap_value(item.next_state)
            targets[row * self.action_size + item.action] = target_value

        target_tensor = Tensor(NdArray(targets, (batch_size, self.action_size)), requires_grad=False)
        loss = self.loss_fn(predictions, target_tensor)
        loss_value = loss.value.data[0]
        loss.backward()
        self.optimizer.step()
        self._decay_epsilon()
        return loss_value

    def sync_target_network(self) -> None:
        """将在线 Q 网络参数同步到 target network。"""
        if self.target_network is None:
            raise ValueError("target_network is not configured")
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _predict_next_q_values(self, state: Sequence[float]) -> List[float]:
        network = self.target_network if self.target_network is not None else self.q_network
        state_values = self._as_state_list(state)
        tensor = Tensor(NdArray(state_values, (1, self.state_dim)), requires_grad=False)
        with no_grad():
            output = network(tensor)
        return list(output.value.data)

    def _bootstrap_value(self, state: Sequence[float]) -> float:
        """计算 Bellman 目标中的下一状态价值。"""
        return max(self._predict_next_q_values(state))

    def _as_state_list(self, state: Sequence[float]) -> List[float]:
        values = list(state)
        if len(values) != self.state_dim:
            raise ValueError(f"state length {len(values)} does not match state_dim {self.state_dim}")
        return [float(value) for value in values]

    def _flatten_states(self, states: Sequence[Sequence[float]]) -> List[float]:
        flat = []
        for state in states:
            flat.extend(self._as_state_list(state))
        return flat

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class DoubleDQNAgent(DQNAgent):
    """Double DQN 智能体。

    Double DQN 使用在线网络选择下一动作，再用目标网络评估该动作价值，
    从而缓解普通 DQN 中 ``max`` 操作带来的过估计。
    """

    def _bootstrap_value(self, state: Sequence[float]) -> float:
        online_values = self.predict_q_values(state)
        best_action = _argmax(online_values)
        target_values = self._predict_next_q_values(state)
        return target_values[best_action]


class PolicyGradientAgent:
    """REINFORCE 策略梯度智能体。

    适用于离散动作空间。策略网络输入状态向量，输出每个动作的 logits。
    """

    def __init__(
        self,
        policy_network: Module,
        optimizer: Optimizer,
        state_dim: int,
        action_size: int,
        gamma: float = 0.99,
        normalize_returns: bool = True,
    ):
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if action_size <= 0:
            raise ValueError("action_size must be positive")

        self.policy_network = policy_network
        self.optimizer = optimizer
        self.state_dim = state_dim
        self.action_size = action_size
        self.gamma = gamma
        self.normalize_returns = normalize_returns
        self.policy_loss_fn = CrossEntropyLoss()

    def action_probs(self, state: Sequence[float]) -> List[float]:
        """返回策略网络在给定状态下的动作概率。"""
        logits = self._policy_logits(state, track_grad=False)
        return _softmax(logits.value.data)

    def select_action(self, state: Sequence[float], return_prob: bool = False):
        """按当前策略采样动作。"""
        probs = self.action_probs(state)
        action = _sample_from_probs(probs)
        if return_prob:
            return action, probs[action]
        return action

    def learn_trajectory(self, trajectory: Sequence[Transition]) -> Optional[float]:
        """基于完整轨迹执行一次 REINFORCE 更新。"""
        if not trajectory:
            return None

        returns = _discounted_returns([item.reward for item in trajectory], self.gamma)
        weights = _normalize(returns) if self.normalize_returns else returns

        self.optimizer.zero_grad()
        total_loss = None
        for item, weight in zip(trajectory, weights):
            logits = self._policy_logits(item.state, track_grad=True)
            loss = self._weighted_action_loss(logits, item.action, weight)
            total_loss = loss if total_loss is None else total_loss + loss

        total_loss = total_loss / len(trajectory)
        loss_value = total_loss.value.data[0]
        total_loss.backward()
        self.optimizer.step()
        return loss_value

    def _policy_logits(self, state: Sequence[float], track_grad: bool) -> Tensor:
        state_values = self._as_state_list(state)
        tensor = Tensor(NdArray(state_values, (1, self.state_dim)), requires_grad=False)
        if track_grad:
            return self.policy_network(tensor)
        with no_grad():
            return self.policy_network(tensor)

    def _weighted_action_loss(self, logits: Tensor, action: int, weight: float) -> Tensor:
        if not 0 <= action < self.action_size:
            raise ValueError(f"Invalid action {action}; expected 0 <= action < {self.action_size}")
        target = Tensor(NdArray([action], (1,), "int32"), requires_grad=False)
        return self.policy_loss_fn(logits, target) * float(weight)

    def _as_state_list(self, state: Sequence[float]) -> List[float]:
        values = list(state)
        if len(values) != self.state_dim:
            raise ValueError(f"state length {len(values)} does not match state_dim {self.state_dim}")
        return [float(value) for value in values]


class ActorCriticAgent(PolicyGradientAgent):
    """Advantage Actor-Critic 智能体。

    policy network 输出动作 logits；value network 输出单个状态价值。
    """

    def __init__(
        self,
        policy_network: Module,
        value_network: Module,
        policy_optimizer: Optimizer,
        value_optimizer: Optimizer,
        state_dim: int,
        action_size: int,
        gamma: float = 0.99,
        normalize_advantages: bool = True,
    ):
        super().__init__(
            policy_network=policy_network,
            optimizer=policy_optimizer,
            state_dim=state_dim,
            action_size=action_size,
            gamma=gamma,
            normalize_returns=False,
        )
        self.value_network = value_network
        self.value_optimizer = value_optimizer
        self.normalize_advantages = normalize_advantages
        self.value_loss_fn = MSELoss()

    def value(self, state: Sequence[float]) -> float:
        """估计状态价值。"""
        prediction = self._value_prediction(state, track_grad=False)
        return prediction.value.data[0]

    def learn_trajectory(self, trajectory: Sequence[Transition]) -> Optional[Dict[str, float]]:
        """基于轨迹执行一次 actor-critic 更新。"""
        if not trajectory:
            return None

        returns = _discounted_returns([item.reward for item in trajectory], self.gamma)
        baseline_values = [self.value(item.state) for item in trajectory]
        advantages = [ret - value for ret, value in zip(returns, baseline_values)]
        policy_weights = _normalize(advantages) if self.normalize_advantages else advantages

        self.optimizer.zero_grad()
        policy_loss = None
        for item, advantage in zip(trajectory, policy_weights):
            logits = self._policy_logits(item.state, track_grad=True)
            loss = self._weighted_action_loss(logits, item.action, advantage)
            policy_loss = loss if policy_loss is None else policy_loss + loss
        policy_loss = policy_loss / len(trajectory)
        policy_loss_value = policy_loss.value.data[0]
        policy_loss.backward()
        self.optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss = None
        for item, ret in zip(trajectory, returns):
            prediction = self._value_prediction(item.state, track_grad=True)
            target = Tensor(NdArray([ret], (1, 1)), requires_grad=False)
            loss = self.value_loss_fn(prediction, target)
            value_loss = loss if value_loss is None else value_loss + loss
        value_loss = value_loss / len(trajectory)
        value_loss_value = value_loss.value.data[0]
        value_loss.backward()
        self.value_optimizer.step()

        return {"policy_loss": policy_loss_value, "value_loss": value_loss_value}

    def _value_prediction(self, state: Sequence[float], track_grad: bool) -> Tensor:
        state_values = self._as_state_list(state)
        tensor = Tensor(NdArray(state_values, (1, self.state_dim)), requires_grad=False)
        if track_grad:
            return self.value_network(tensor)
        with no_grad():
            return self.value_network(tensor)


class PPOAgent(ActorCriticAgent):
    """Proximal Policy Optimization 智能体。

    这是面向 tinyTorch 的简化 PPO：使用旧策略概率计算 clipped 权重，
    再以加权策略交叉熵更新 actor，同时用 MSE 更新 critic。
    """

    def __init__(
        self,
        policy_network: Module,
        value_network: Module,
        policy_optimizer: Optimizer,
        value_optimizer: Optimizer,
        state_dim: int,
        action_size: int,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        normalize_advantages: bool = True,
    ):
        super().__init__(
            policy_network=policy_network,
            value_network=value_network,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            state_dim=state_dim,
            action_size=action_size,
            gamma=gamma,
            normalize_advantages=normalize_advantages,
        )
        if clip_epsilon <= 0:
            raise ValueError("clip_epsilon must be positive")
        self.clip_epsilon = clip_epsilon

    def learn_trajectory(
        self,
        trajectory: Sequence[Transition],
        old_action_probs: Sequence[float],
    ) -> Optional[Dict[str, float]]:
        """基于轨迹和采样时的动作概率执行一次 PPO 更新。"""
        if not trajectory:
            return None
        if len(old_action_probs) != len(trajectory):
            raise ValueError("old_action_probs length must match trajectory length")

        returns = _discounted_returns([item.reward for item in trajectory], self.gamma)
        baseline_values = [self.value(item.state) for item in trajectory]
        advantages = [ret - value for ret, value in zip(returns, baseline_values)]
        if self.normalize_advantages:
            advantages = _normalize(advantages)

        self.optimizer.zero_grad()
        policy_loss = None
        for item, old_prob, advantage in zip(trajectory, old_action_probs, advantages):
            if old_prob <= 0:
                raise ValueError("old_action_probs must be positive")
            new_prob = self.action_probs(item.state)[item.action]
            ratio = new_prob / old_prob
            clipped_ratio = min(max(ratio, 1.0 - self.clip_epsilon), 1.0 + self.clip_epsilon)
            if advantage >= 0:
                weight = min(ratio * advantage, clipped_ratio * advantage)
            else:
                weight = max(ratio * advantage, clipped_ratio * advantage)
            logits = self._policy_logits(item.state, track_grad=True)
            loss = self._weighted_action_loss(logits, item.action, weight)
            policy_loss = loss if policy_loss is None else policy_loss + loss
        policy_loss = policy_loss / len(trajectory)
        policy_loss_value = policy_loss.value.data[0]
        policy_loss.backward()
        self.optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss = None
        for item, ret in zip(trajectory, returns):
            prediction = self._value_prediction(item.state, track_grad=True)
            target = Tensor(NdArray([ret], (1, 1)), requires_grad=False)
            loss = self.value_loss_fn(prediction, target)
            value_loss = loss if value_loss is None else value_loss + loss
        value_loss = value_loss / len(trajectory)
        value_loss_value = value_loss.value.data[0]
        value_loss.backward()
        self.value_optimizer.step()

        return {"policy_loss": policy_loss_value, "value_loss": value_loss_value}
