"""随机梯度下降（SGD）优化器。

Author: TinyAI Team
"""

from typing import List
from tinytorch.nn.parameter import Parameter
from tinytorch.ml.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    """随机梯度下降优化器。
    
    支持动量和权重衰减。
    
    更新规则（不带动量）:
        param = param - learning_rate * (grad + weight_decay * param)
    
    更新规则（带动量）:
        velocity = momentum * velocity + grad + weight_decay * param
        param = param - learning_rate * velocity
    
    Attributes:
        params: 要优化的参数列表
        learning_rate: 学习率
        momentum: 动量系数，默认为 0（不使用动量）
        weight_decay: 权重衰减系数，默认为 0（不使用权重衰减）
    
    Example:
        >>> optimizer = SGD(model.parameters(), learning_rate=0.01, momentum=0.9)
        >>> for epoch in range(num_epochs):
        ...     optimizer.zero_grad()
        ...     loss = compute_loss()
        ...     loss.backward()
        ...     optimizer.step()
    """
    
    def __init__(self, params: List[Parameter], learning_rate: float = 0.01,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        """初始化 SGD 优化器。
        
        Args:
            params: 要优化的参数列表
            learning_rate: 学习率
            momentum: 动量系数（0-1之间）
            weight_decay: 权重衰减系数（L2 正则化）
        """
        super().__init__(params, learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # 初始化动量缓冲区
        if momentum > 0:
            for i, param in enumerate(self.params):
                self.state[i] = {'velocity': None}
    
    def step(self) -> None:
        """执行一步参数更新。"""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            # 获取梯度（grad 是 Tensor 对象）
            grad = param.grad
            
            # 添加权重衰减（L2 正则化）
            if self.weight_decay > 0:
                # grad = grad + weight_decay * param
                for j in range(len(grad.data)):
                    grad.data[j] += self.weight_decay * param.value.data[j]
            
            # 使用动量
            if self.momentum > 0:
                if self.state[i]['velocity'] is None:
                    # 初始化速度为零
                    from tinytorch.tensor.tensor import Tensor
                    self.state[i]['velocity'] = Tensor.zeros(param.value.shape.dims)
                
                velocity = self.state[i]['velocity']
                
                # velocity = momentum * velocity + grad
                for j in range(len(velocity.data)):
                    velocity.data[j] = self.momentum * velocity.data[j] + grad.data[j]
                
                # param = param - learning_rate * velocity
                for j in range(len(param.value.data)):
                    param.value.data[j] -= self.learning_rate * velocity.data[j]
            else:
                # 普通 SGD: param = param - learning_rate * grad
                for j in range(len(param.value.data)):
                    param.value.data[j] -= self.learning_rate * grad.data[j]
    
    def __repr__(self) -> str:
        """返回优化器的字符串表示。"""
        return (f"SGD(learning_rate={self.learning_rate}, "
                f"momentum={self.momentum}, weight_decay={self.weight_decay})")
