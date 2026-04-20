"""Adam 优化器。

Author: TinyAI Team
"""

import math
from typing import List
from tinytorch.ml.optimizers.optimizer import Optimizer
from tinytorch.nn.parameter import Parameter


class Adam(Optimizer):
    """Adam (Adaptive Moment Estimation) 优化器。
    
    Adam 结合了 RMSprop 和 Momentum 的优点，使用一阶矩估计和二阶矩估计
    自适应地调整每个参数的学习率。
    
    更新公式：
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)
    
    Attributes:
        learning_rate: 学习率
        beta1: 一阶矩估计的指数衰减率
        beta2: 二阶矩估计的指数衰减率
        epsilon: 数值稳定性常数
        weight_decay: 权重衰减（L2 正则化）
    
    Example:
        >>> optimizer = Adam(model.parameters(), learning_rate=0.001)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """
    
    def __init__(self, parameters: List[Parameter], learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8,
                 weight_decay: float = 0.0, **kwargs):
        """初始化 Adam 优化器。
        
        Args:
            parameters: 待优化的参数列表
            learning_rate: 学习率（也可用 lr 关键字）
            beta1: 一阶矩估计的指数衰减率（也可用 betas 元组）
            beta2: 二阶矩估计的指数衰减率（也可用 betas 元组）
            epsilon: 数值稳定性常数
            weight_decay: 权重衰减系数
        """
        # 支持 lr 作为 learning_rate 的别名
        if 'lr' in kwargs:
            learning_rate = kwargs.pop('lr')
        # 支持 betas 元组参数
        if 'betas' in kwargs:
            betas = kwargs.pop('betas')
            beta1, beta2 = betas[0], betas[1]
        super().__init__(parameters, learning_rate)
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # 初始化一阶矩和二阶矩
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        self.t = 0   # 时间步
        
        for param in parameters:
            param_id = id(param)
            # 初始化为零
            self.m[param_id] = [0.0] * len(param.value.data)
            self.v[param_id] = [0.0] * len(param.value.data)
    
    def step(self):
        """执行一步参数更新。

        对一阶矩/二阶矩的更新仍使用列表操作（需要原地修改状态），
        但最终参数更新使用 NdArray 向量化运算。
        """
        self.t += 1

        for param in self.params:
            if param.grad is None:
                continue

            param_id = id(param)
            grad_data = param.grad.data

            # 权重衰减（L2 正则化）
            if self.weight_decay != 0:
                param_data = param.value.data
                grad_data = [
                    grad_data[i] + self.weight_decay * param_data[i]
                    for i in range(len(grad_data))
                ]

            # 更新一阶矩和二阶矩
            m_list = self.m[param_id]
            v_list = self.v[param_id]
            one_minus_beta1 = 1 - self.beta1
            one_minus_beta2 = 1 - self.beta2
            for i in range(len(grad_data)):
                m_list[i] = self.beta1 * m_list[i] + one_minus_beta1 * grad_data[i]
                v_list[i] = self.beta2 * v_list[i] + one_minus_beta2 * grad_data[i] * grad_data[i]

            # 偏差修正
            bias_correction1 = 1 - self.beta1 ** self.t
            bias_correction2 = 1 - self.beta2 ** self.t

            # 向量化参数更新
            lr_scaled = self.learning_rate / bias_correction1
            bc2_inv = 1.0 / bias_correction2
            eps = self.epsilon
            param.value.data = [
                param.value.data[i] - lr_scaled * m_list[i] / (math.sqrt(v_list[i] * bc2_inv) + eps)
                for i in range(len(param.value.data))
            ]
    
    def __repr__(self) -> str:
        """返回优化器的字符串表示。"""
        return (f"Adam(learning_rate={self.learning_rate}, beta1={self.beta1}, "
                f"beta2={self.beta2}, epsilon={self.epsilon}, weight_decay={self.weight_decay})")
