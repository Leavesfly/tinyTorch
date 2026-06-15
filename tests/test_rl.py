"""强化学习模块测试。"""

import pytest

from tinytorch.ml.optimizers import SGD
from tinytorch.nn import Linear, Sequential
from tinytorch.rl import (
    ActorCriticAgent,
    Box,
    DQNAgent,
    Discrete,
    DoubleDQNAgent,
    GridWorldEnv,
    MultiDiscrete,
    PPOAgent,
    PolicyGradientAgent,
    PrioritizedReplayBuffer,
    QLearningAgent,
    ReplayBuffer,
    RolloutBuffer,
    Transition,
    VectorEnv,
)
from tinytorch.utils import random as tt_random


def test_discrete_space_sample_and_contains():
    tt_random.seed(42)
    space = Discrete(4)

    samples = [space.sample() for _ in range(20)]

    assert all(space.contains(sample) for sample in samples)
    assert not space.contains(-1)
    assert not space.contains(4)


def test_box_and_multidiscrete_spaces():
    tt_random.seed(42)
    box = Box(low=-1.0, high=[1.0, 2.0], shape=(2,))
    sample = box.sample()
    multi = MultiDiscrete([2, 3])
    multi_sample = multi.sample()

    assert box.contains(sample)
    assert not box.contains([2.0, 0.0])
    assert multi.contains(multi_sample)
    assert not multi.contains([2, 0])


def test_gridworld_step_reaches_goal():
    env = GridWorldEnv(width=2, height=2, start=(0, 0), goal=(1, 0), max_steps=4)

    state = env.reset()
    next_state, reward, done, info = env.step(1)

    assert state == (0, 0)
    assert next_state == (1, 0)
    assert reward == pytest.approx(1.0)
    assert done
    assert info["reached_goal"]


def test_gridworld_blocks_walls_and_obstacles():
    env = GridWorldEnv(width=3, height=3, start=(0, 0), goal=(2, 2), obstacles=[(1, 0)])

    next_state, reward, done, info = env.step(1)

    assert next_state == (0, 0)
    assert reward == pytest.approx(env.wall_reward)
    assert not done
    assert info["hit_wall"]


def test_vector_env_steps_multiple_envs():
    envs = VectorEnv([
        lambda: GridWorldEnv(width=2, height=1, start=(0, 0), goal=(1, 0)),
        lambda: GridWorldEnv(width=2, height=1, start=(0, 0), goal=(1, 0)),
    ])

    states = envs.reset()
    next_states, rewards, dones, infos = envs.step([1, 1])

    assert states == [(0, 0), (0, 0)]
    assert next_states == [(1, 0), (1, 0)]
    assert rewards == [1.0, 1.0]
    assert dones == [True, True]
    assert all(info["reached_goal"] for info in infos)


def test_replay_buffer_capacity_and_sample():
    tt_random.seed(42)
    buffer = ReplayBuffer(capacity=2)

    buffer.push([0.0, 0.0], 0, 0.0, [0.0, 1.0], False)
    buffer.push([0.0, 1.0], 1, 0.5, [1.0, 1.0], False)
    buffer.push([1.0, 1.0], 2, 1.0, [1.0, 1.0], True)

    assert len(buffer) == 2
    sample = buffer.sample(1)
    assert len(sample) == 1
    assert isinstance(sample[0], Transition)


def test_prioritized_replay_buffer_samples_and_updates_priorities():
    tt_random.seed(42)
    buffer = PrioritizedReplayBuffer(capacity=4)

    buffer.push([0.0, 0.0], 0, 0.0, [0.0, 1.0], False, priority=1.0)
    buffer.push([0.0, 1.0], 1, 1.0, [1.0, 1.0], True, priority=2.0)
    sample = buffer.sample(2)
    buffer.update_priorities(sample.indices, [0.5, 1.5])

    assert len(sample.transitions) == 2
    assert len(sample.indices) == 2
    assert all(weight > 0.0 for weight in sample.weights)


def test_rollout_buffer_computes_gae_returns():
    rollout = RolloutBuffer()
    rollout.add([0.0, 0.0], 0, 1.0, [1.0, 0.0], False, value=0.2, log_prob=0.0)
    rollout.add([1.0, 0.0], 1, 0.5, [1.0, 1.0], True, value=0.1, log_prob=0.0)

    rollout.compute_returns_and_advantages(gamma=0.9, gae_lambda=1.0, normalize_advantages=False)

    assert len(rollout.returns) == 2
    assert rollout.returns[0] == pytest.approx(1.45)
    assert rollout.returns[1] == pytest.approx(0.5)
    assert len(rollout.to_transitions()) == 2


def test_q_learning_update_improves_rewarded_action():
    agent = QLearningAgent(action_size=2, learning_rate=0.5, gamma=0.9, epsilon=0.0)

    td_error = agent.update("s0", action=1, reward=1.0, next_state="s1", done=True)

    assert td_error == pytest.approx(1.0)
    assert agent.q_values("s0")[1] == pytest.approx(0.5)
    assert agent.select_action("s0", training=False) == 1


def test_q_learning_train_episode_runs():
    tt_random.seed(42)
    env = GridWorldEnv(width=2, height=1, start=(0, 0), goal=(1, 0), max_steps=4)
    agent = QLearningAgent(action_size=env.action_space.n, learning_rate=1.0, epsilon=0.0)

    total_reward, steps = agent.train_episode(env)

    assert steps == 2
    assert total_reward == pytest.approx(0.9)


def test_dqn_agent_learns_one_step():
    tt_random.seed(42)
    q_network = Sequential(Linear(2, 4), Linear(4, 2))
    optimizer = SGD(q_network.parameters(), learning_rate=0.05)
    agent = DQNAgent(
        q_network=q_network,
        optimizer=optimizer,
        state_dim=2,
        action_size=2,
        gamma=0.9,
        epsilon=0.2,
        epsilon_decay=0.5,
        batch_size=1,
    )

    agent.remember([0.0, 0.0], action=1, reward=1.0, next_state=[1.0, 0.0], done=True)
    loss = agent.learn()

    assert loss is not None
    assert loss >= 0.0
    assert agent.epsilon == pytest.approx(0.1)
    assert agent.select_action([0.0, 0.0], training=False) in (0, 1)


def test_dqn_train_step_and_soft_target_update():
    tt_random.seed(42)
    q_network = Sequential(Linear(2, 4), Linear(4, 2))
    target_network = Sequential(Linear(2, 4), Linear(4, 2))
    optimizer = SGD(q_network.parameters(), learning_rate=0.05)
    agent = DQNAgent(
        q_network=q_network,
        target_network=target_network,
        optimizer=optimizer,
        state_dim=2,
        action_size=2,
        epsilon=0.0,
        batch_size=1,
    )
    env = GridWorldEnv(width=2, height=1, start=(0, 0), goal=(1, 0))

    next_state, reward, done, loss = agent.train_step(env, state=(0, 0))
    agent.soft_update_target_network(tau=0.5)

    assert next_state in [(0, 0), (1, 0)]
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert loss is not None


def test_double_dqn_agent_learns_with_target_network():
    tt_random.seed(42)
    q_network = Sequential(Linear(2, 4), Linear(4, 2))
    target_network = Sequential(Linear(2, 4), Linear(4, 2))
    optimizer = SGD(q_network.parameters(), learning_rate=0.05)
    agent = DoubleDQNAgent(
        q_network=q_network,
        target_network=target_network,
        optimizer=optimizer,
        state_dim=2,
        action_size=2,
        gamma=0.9,
        epsilon=0.0,
        batch_size=1,
    )

    agent.remember([0.0, 0.0], action=0, reward=0.5, next_state=[1.0, 0.0], done=False)
    loss = agent.learn()

    assert loss is not None
    assert loss >= 0.0
    assert agent.select_action([0.0, 0.0], training=False) in (0, 1)


def test_policy_gradient_agent_learns_trajectory():
    tt_random.seed(42)
    policy_network = Sequential(Linear(2, 4), Linear(4, 2))
    optimizer = SGD(policy_network.parameters(), learning_rate=0.05)
    agent = PolicyGradientAgent(
        policy_network=policy_network,
        optimizer=optimizer,
        state_dim=2,
        action_size=2,
        gamma=0.9,
        normalize_returns=False,
    )
    trajectory = [
        Transition([0.0, 0.0], 0, 1.0, [1.0, 0.0], False),
        Transition([1.0, 0.0], 1, 0.5, [1.0, 1.0], True),
    ]

    action, prob = agent.select_action([0.0, 0.0], return_prob=True)
    loss = agent.learn_trajectory(trajectory)

    assert action in (0, 1)
    assert 0.0 < prob <= 1.0
    assert loss is not None


def test_actor_critic_agent_learns_trajectory():
    tt_random.seed(42)
    policy_network = Sequential(Linear(2, 4), Linear(4, 2))
    value_network = Sequential(Linear(2, 4), Linear(4, 1))
    agent = ActorCriticAgent(
        policy_network=policy_network,
        value_network=value_network,
        policy_optimizer=SGD(policy_network.parameters(), learning_rate=0.05),
        value_optimizer=SGD(value_network.parameters(), learning_rate=0.05),
        state_dim=2,
        action_size=2,
        gamma=0.9,
        normalize_advantages=False,
    )
    trajectory = [
        Transition([0.0, 0.0], 0, 1.0, [1.0, 0.0], False),
        Transition([1.0, 0.0], 1, 0.5, [1.0, 1.0], True),
    ]

    metrics = agent.learn_trajectory(trajectory)

    assert metrics is not None
    assert "policy_loss" in metrics
    assert metrics["value_loss"] >= 0.0


def test_ppo_agent_learns_trajectory():
    tt_random.seed(42)
    policy_network = Sequential(Linear(2, 4), Linear(4, 2))
    value_network = Sequential(Linear(2, 4), Linear(4, 1))
    agent = PPOAgent(
        policy_network=policy_network,
        value_network=value_network,
        policy_optimizer=SGD(policy_network.parameters(), learning_rate=0.05),
        value_optimizer=SGD(value_network.parameters(), learning_rate=0.05),
        state_dim=2,
        action_size=2,
        gamma=0.9,
        clip_epsilon=0.2,
        normalize_advantages=False,
    )
    trajectory = [
        Transition([0.0, 0.0], 0, 1.0, [1.0, 0.0], False),
        Transition([1.0, 0.0], 1, 0.5, [1.0, 1.0], True),
    ]
    old_action_probs = [0.5, 0.5]

    metrics = agent.learn_trajectory(trajectory, old_action_probs)

    assert metrics is not None
    assert "policy_loss" in metrics
    assert metrics["value_loss"] >= 0.0


def test_ppo_agent_learns_rollout_buffer():
    tt_random.seed(42)
    policy_network = Sequential(Linear(2, 4), Linear(4, 2))
    value_network = Sequential(Linear(2, 4), Linear(4, 1))
    agent = PPOAgent(
        policy_network=policy_network,
        value_network=value_network,
        policy_optimizer=SGD(policy_network.parameters(), learning_rate=0.05),
        value_optimizer=SGD(value_network.parameters(), learning_rate=0.05),
        state_dim=2,
        action_size=2,
        gamma=0.9,
        clip_epsilon=0.2,
        normalize_advantages=False,
    )
    rollout = RolloutBuffer()
    rollout.add([0.0, 0.0], 0, 1.0, [1.0, 0.0], False, value=0.0, log_prob=0.0)
    rollout.add([1.0, 0.0], 1, 0.5, [1.0, 1.0], True, value=0.0, log_prob=0.0)
    rollout.compute_returns_and_advantages(gamma=0.9, normalize_advantages=False)

    metrics = agent.learn_rollout(rollout)

    assert metrics is not None
    assert "policy_loss" in metrics
    assert metrics["value_loss"] >= 0.0
