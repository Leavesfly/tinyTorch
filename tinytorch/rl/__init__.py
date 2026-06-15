"""强化学习模块。

提供教学友好的强化学习基础组件：离散空间、GridWorld 环境、经验回放、
表格 Q-learning 以及主流深度强化学习算法。
"""

from tinytorch.rl.agents import ActorCriticAgent, DQNAgent, DoubleDQNAgent, PolicyGradientAgent, PPOAgent, QLearningAgent
from tinytorch.rl.envs import Env, GridWorldEnv, VectorEnv
from tinytorch.rl.replay import PrioritizedReplayBuffer, PrioritizedSample, ReplayBuffer, RolloutBuffer, RolloutStep, Transition
from tinytorch.rl.spaces import Box, Discrete, MultiDiscrete, Space

__all__ = [
    "DQNAgent",
    "DoubleDQNAgent",
    "PolicyGradientAgent",
    "ActorCriticAgent",
    "PPOAgent",
    "QLearningAgent",
    "Env",
    "GridWorldEnv",
    "VectorEnv",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "PrioritizedSample",
    "RolloutBuffer",
    "RolloutStep",
    "Transition",
    "Space",
    "Discrete",
    "Box",
    "MultiDiscrete",
]
