"""均方误差损失函数。

Author: TinyAI Team
"""

from tinytorch.autograd.tensor import Tensor
from tinytorch.ndarr.ndarray import NdArray
from tinytorch.ndarr.shape import Shape
from tinytorch.ml.losses.loss import Loss


class MSELoss(Loss):
    """均方误差损失（Mean Squared Error Loss）。
    
    计算预测值和目标值之间的均方误差。
    
    公式:
        reduction='mean': loss = mean((pred - target)^2)
        reduction='sum':  loss = sum((pred - target)^2)
        reduction='none': loss = (pred - target)^2  (逐元素)
    
    主要用于回归任务。
    
    Example:
        >>> loss_fn = MSELoss()
        >>> pred = Tensor(NdArray([[1.0, 2.0], [3.0, 4.0]]))
        >>> target = Tensor(NdArray([[1.5, 2.5], [3.5, 4.5]]))
        >>> loss = loss_fn(pred, target)
        >>> print(loss.value.data[0])  # 约为 0.25
    """
    
    def __init__(self, reduction: str = 'mean'):
        """初始化 MSE 损失函数。
        
        Args:
            reduction: 损失聚合方式，'mean'、'sum' 或 'none'
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction}")
        self.reduction = reduction
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """计算均方误差损失。
        
        Args:
            pred: 预测值
            target: 目标值
        
        Returns:
            损失值
        """
        # 计算差值: diff = pred - target
        diff = pred - target
        
        # 计算平方: squared = diff^2
        squared = diff * diff
        
        if self.reduction == 'mean':
            return squared.mean()
        elif self.reduction == 'sum':
            return squared.sum()
        else:  # 'none' - 返回逐元素损失
            return squared
