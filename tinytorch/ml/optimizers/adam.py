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
                 weight_decay: float = 0.0):
        """初始化 Adam 优化器。
        
        Args:
            parameters: 待优化的参数列表
            learning_rate: 学习率
            beta1: 一阶矩估计的指数衰减率
            beta2: 二阶矩估计的指数衰减率
            epsilon: 数值稳定性常数
            weight_decay: 权重衰减系数
        """
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
        """执行一步参数更新。"""
        self.t += 1
        
        for param in self.parameters:
            if param.grad is None:
                continue
            
            param_id = id(param)
            grad = param.grad  # 获取梯度（grad 是 Tensor 对象）
            
            # 权重衰减
            if self.weight_decay != 0:
                for i in range(len(grad.data)):
                    grad.data[i] += self.weight_decay * param.value.data[i]
            
            # 更新一阶矩和二阶矩
            for i in range(len(param.value.data)):
                # m_t = beta1 * m_{t-1} + (1 - beta1) * grad
                self.m[param_id][i] = self.beta1 * self.m[param_id][i] + (1 - self.beta1) * grad.data[i]
                
                # v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
                self.v[param_id][i] = self.beta2 * self.v[param_id][i] + (1 - self.beta2) * grad.data[i] ** 2
            
            # 偏差修正
            bias_correction1 = 1 - self.beta1 ** self.t
            bias_correction2 = 1 - self.beta2 ** self.t
            
            # 更新参数
            for i in range(len(param.value.data)):
                # m_hat = m_t / (1 - beta1^t)
                m_hat = self.m[param_id][i] / bias_correction1
                
                # v_hat = v_t / (1 - beta2^t)
                v_hat = self.v[param_id][i] / bias_correction2
                
                # param = param - lr * m_hat / (sqrt(v_hat) + eps)
                param.value.data[i] -= self.learning_rate * m_hat / (math.sqrt(v_hat) + self.epsilon)
    
    def __repr__(self) -> str:
        """返回优化器的字符串表示。"""
        return (f"Adam(learning_rate={self.learning_rate}, beta1={self.beta1}, "
                f"beta2={self.beta2}, epsilon={self.epsilon}, weight_decay={self.weight_decay})")
