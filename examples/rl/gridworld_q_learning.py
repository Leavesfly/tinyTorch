"""GridWorld Q-learning 示例。

演示如何使用 ``tinytorch.rl`` 在小型离散环境中训练一个表格强化学习智能体。
"""

from tinytorch.rl import GridWorldEnv, QLearningAgent
from tinytorch.utils import random as tt_random


ACTION_NAMES = ["↑", "→", "↓", "←"]


def format_policy(env: GridWorldEnv, agent: QLearningAgent) -> str:
    """将学到的 greedy policy 格式化为文本。"""
    rows = []
    for y in range(env.height):
        row = []
        for x in range(env.width):
            state = (x, y)
            if state == env.goal:
                row.append("G")
            elif state in env.obstacles:
                row.append("#")
            else:
                action = agent.select_action(state, training=False)
                row.append(ACTION_NAMES[action])
        rows.append(" ".join(row))
    return "\n".join(rows)


def main():
    """运行示例。"""
    tt_random.seed(42)

    env = GridWorldEnv(
        width=4,
        height=4,
        start=(0, 0),
        goal=(3, 3),
        obstacles=[(1, 1), (2, 1)],
        max_steps=40,
    )
    agent = QLearningAgent(
        action_size=env.action_space.n,
        learning_rate=0.3,
        gamma=0.95,
        epsilon=0.8,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    )

    episodes = 200
    rewards = []
    for episode in range(episodes):
        total_reward, steps = agent.train_episode(env)
        rewards.append(total_reward)
        if (episode + 1) % 50 == 0:
            recent_reward = sum(rewards[-50:]) / 50
            print(f"Episode {episode + 1:03d}: avg_reward={recent_reward:.3f}, steps={steps}, epsilon={agent.epsilon:.3f}")

    print("\nFinal greedy policy:")
    print(format_policy(env, agent))

    print("\nQ values at start:")
    print(agent.q_values(env.start))


if __name__ == "__main__":
    main()
