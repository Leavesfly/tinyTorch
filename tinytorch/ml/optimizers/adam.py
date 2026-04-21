"""Adam 优化器。

Author: TinyAI Team
"""

from typing import List
from tinytorch.ml.optimizers.optimizer import Optimizer
from tinytorch.nn.parameter import Parameter
from tinytorch.ndarr import NdArray

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
    
    def __init__(self, params: List[Parameter], learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8,
                 weight_decay: float = 0.0, **kwargs):
        """初始化 Adam 优化器。

        Args:
            params: 待优化的参数列表
            learning_rate: 学习率（也可用 lr 关键字）
            beta1: 一阶矩估计的指数衰减率（也可用 betas 元组）
            beta2: 二阶矩估计的指数衰减率（也可用 betas 元组）
            epsilon: 数值稳定性常数
            weight_decay: 权重衰减系数
        """
        # 支持 betas 元组参数
        if 'betas' in kwargs:
            betas = kwargs.pop('betas')
            beta1, beta2 = betas[0], betas[1]
        super().__init__(params, learning_rate, **kwargs)
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # 初始化一阶矩和二阶矩
        self.moment_first = {}  # 一阶矩估计
        self.moment_second = {}  # 二阶矩估计
        self.timestep = 0   # 时间步
        
        for param in params:
            param_id = id(param)
            # 初始化为零 NdArray
            self.moment_first[param_id] = NdArray.zeros(param.value.shape, param.value.dtype)
            self.moment_second[param_id] = NdArray.zeros(param.value.shape, param.value.dtype)
    
    def step(self):
        """执行一步参数更新。

        全部使用 NdArray 向量化运算，不再使用任何 Python for 循环遍历元素。
        """
        self.timestep += 1

        for param in self.params:
            if param.grad is None:
                continue

            param_id = id(param)
            
            # 将梯度构造为 NdArray
            grad = NdArray(param.grad.data, param.grad.shape, param.grad.dtype)

            # 权重衰减（L2 正则化）
            if self.weight_decay != 0:
                grad = grad.add(param.value.mul(self.weight_decay))

            # 更新一阶矩和二阶矩（向量化运算）
            one_minus_beta1 = 1 - self.beta1
            one_minus_beta2 = 1 - self.beta2
            
            self.moment_first[param_id] = self.moment_first[param_id].mul(self.beta1).add(grad.mul(one_minus_beta1))
            self.moment_second[param_id] = self.moment_second[param_id].mul(self.beta2).add(grad.mul(grad).mul(one_minus_beta2))

            # 偏差修正
            bias_correction1 = 1 - self.beta1 ** self.timestep
            bias_correction2 = 1 - self.beta2 ** self.timestep

            # 计算修正后的矩估计
            m_hat = self.moment_first[param_id].div(bias_correction1)
            v_hat = self.moment_second[param_id].div(bias_correction2)

            # 参数更新（向量化运算）
            update = m_hat.div(v_hat.sqrt().add(self.epsilon)).mul(self.learning_rate)
            param.value = param.value.sub(update)
    
    def __repr__(self) -> str:
        """返回优化器的字符串表示。"""
        return (f"Adam(learning_rate={self.learning_rate}, beta1={self.beta1}, "
                f"beta2={self.beta2}, epsilon={self.epsilon}, weight_decay={self.weight_decay})")