"""均方误差损失函数。

Author: TinyAI Team
"""

from tinytorch.autograd.variable import Variable
from tinytorch.ml.losses.loss import Loss


class MSELoss(Loss):
    """均方误差损失（Mean Squared Error Loss）。
    
    计算预测值和目标值之间的均方误差。
    
    公式: loss = mean((pred - target)^2)
    
    主要用于回归任务。
    
    Example:
        >>> loss_fn = MSELoss()
        >>> pred = Variable(Tensor([[1.0, 2.0], [3.0, 4.0]]))
        >>> target = Variable(Tensor([[1.5, 2.5], [3.5, 4.5]]))
        >>> loss = loss_fn(pred, target)
        >>> print(loss.value.data[0])  # 约为 0.25
    """
    
    def __init__(self):
        """初始化 MSE 损失函数。"""
        super().__init__()
    
    def forward(self, pred: Variable, target: Variable) -> Variable:
        """计算均方误差损失。
        
        Args:
            pred: 预测值
            target: 目标值
        
        Returns:
            损失值（标量）
        """
        # 计算差值: diff = pred - target
        diff = pred - target
        
        # 计算平方: squared = diff^2
        squared = diff * diff
        
        # 计算均值: loss = mean(squared)
        loss = squared.mean()
        
        return loss
