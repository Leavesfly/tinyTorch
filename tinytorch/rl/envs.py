"""强化学习环境。

这些环境遵循简化版 Gym 风格接口：``reset()`` 和 ``step(action)``。
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from tinytorch.rl.spaces import Discrete


GridState = Tuple[int, int]


class Env:
    """强化学习环境基类。"""

    action_space = None
    observation_space = None

    def reset(self):
        """重置环境并返回初始观测。"""
        raise NotImplementedError

    def step(self, action):
        """执行一步环境转移，返回 ``(obs, reward, done, info)``。"""
        raise NotImplementedError

    def render(self) -> str:
        """返回环境的可读表示。"""
        return repr(self)


class VectorEnv:
    """顺序向量环境。

    该实现不引入多进程，只是在一个对象中管理多个环境实例，便于教学和小规模采样。
    """

    def __init__(self, env_fns: Sequence[Callable[[], Env]]):
        if not env_fns:
            raise ValueError("VectorEnv requires at least one environment factory")
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

    def reset(self) -> List[object]:
        """重置所有环境。"""
        return [env.reset() for env in self.envs]

    def step(self, actions: Sequence[object]):
        """对所有环境各执行一步。"""
        if len(actions) != self.num_envs:
            raise ValueError("actions length must match number of environments")
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        observations, rewards, dones, infos = zip(*results)
        return list(observations), list(rewards), list(dones), list(infos)


class GridWorldEnv(Env):
    """二维网格世界环境。

    动作定义：
        0: 上，1: 右，2: 下，3: 左

    Args:
        width: 网格宽度
        height: 网格高度
        start: 起点坐标 ``(x, y)``
        goal: 终点坐标 ``(x, y)``
        obstacles: 障碍物坐标集合，智能体无法进入
        step_reward: 普通移动奖励
        goal_reward: 到达终点奖励
        wall_reward: 撞墙或撞障碍物奖励
        max_steps: 单个 episode 的最大步数
    """

    ACTIONS: Tuple[GridState, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))

    def __init__(
        self,
        width: int = 4,
        height: int = 4,
        start: GridState = (0, 0),
        goal: Optional[GridState] = None,
        obstacles: Optional[Sequence[GridState]] = None,
        step_reward: float = -0.01,
        goal_reward: float = 1.0,
        wall_reward: float = -0.1,
        max_steps: Optional[int] = None,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("GridWorld width and height must be positive")

        self.width = width
        self.height = height
        self.start = start
        self.goal = goal if goal is not None else (width - 1, height - 1)
        self.obstacles = set(obstacles or [])
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.wall_reward = wall_reward
        self.max_steps = max_steps if max_steps is not None else width * height * 4

        self.action_space = Discrete(4)
        self.observation_space = Discrete(width * height)
        self.state = self.start
        self.steps = 0

        self._validate_position(self.start, "start")
        self._validate_position(self.goal, "goal")
        for obstacle in self.obstacles:
            self._validate_position(obstacle, "obstacle")
        if self.start in self.obstacles or self.goal in self.obstacles:
            raise ValueError("start and goal cannot be obstacles")

    def _validate_position(self, position: GridState, name: str) -> None:
        x, y = position
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"{name} position {position} is outside the grid")

    def reset(self) -> GridState:
        """重置环境并返回初始状态。"""
        self.state = self.start
        self.steps = 0
        return self.state

    def step(self, action: int) -> Tuple[GridState, float, bool, Dict[str, object]]:
        """执行一步环境转移。"""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0 <= action < {self.action_space.n}")

        self.steps += 1
        dx, dy = self.ACTIONS[action]
        x, y = self.state
        candidate = (x + dx, y + dy)
        reward = self.step_reward
        hit_wall = False

        if not self._is_free(candidate):
            candidate = self.state
            reward = self.wall_reward
            hit_wall = True

        self.state = candidate
        reached_goal = self.state == self.goal
        timed_out = self.steps >= self.max_steps
        done = reached_goal or timed_out

        if reached_goal:
            reward = self.goal_reward

        info = {
            "hit_wall": hit_wall,
            "reached_goal": reached_goal,
            "timed_out": timed_out,
            "steps": self.steps,
        }
        return self.state, reward, done, info

    def _is_free(self, position: GridState) -> bool:
        x, y = position
        in_bounds = 0 <= x < self.width and 0 <= y < self.height
        return in_bounds and position not in self.obstacles

    def state_index(self, state: Optional[GridState] = None) -> int:
        """将 ``(x, y)`` 状态映射为离散索引。"""
        x, y = state if state is not None else self.state
        return y * self.width + x

    def state_vector(self, state: Optional[GridState] = None) -> List[float]:
        """返回归一化坐标向量，便于 DQN 作为输入。"""
        x, y = state if state is not None else self.state
        x_scale = max(1, self.width - 1)
        y_scale = max(1, self.height - 1)
        return [x / x_scale, y / y_scale]

    def render(self) -> str:
        """返回文本形式的网格。"""
        rows = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)
                if pos == self.state:
                    row.append("A")
                elif pos == self.goal:
                    row.append("G")
                elif pos in self.obstacles:
                    row.append("#")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)
