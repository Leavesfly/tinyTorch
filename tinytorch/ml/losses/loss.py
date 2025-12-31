"""损失函数基类。

Author: TinyAI Team
"""

from tinytorch.autograd.variable import Variable


class Loss:
    """损失函数基类。
    
    所有损失函数都应该继承这个类。子类需要实现 forward 方法定义损失计算逻辑。
    
    Example:
        >>> loss_fn = MSELoss()
        >>> pred = model(input)
        >>> loss = loss_fn(pred, target)
        >>> loss.backward()
    """
    
    def __init__(self, name: str = None):
        """初始化损失函数。
        
        Args:
            name: 损失函数名称
        """
        self.name = name or self.__class__.__name__
    
    def forward(self, pred: Variable, target: Variable) -> Variable:
        """计算损失（抽象方法，子类必须实现）。
        
        Args:
            pred: 预测值
            target: 目标值
        
        Returns:
            损失值
        
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward method")
    
    def __call__(self, pred: Variable, target: Variable) -> Variable:
        """调用损失函数。
        
        Args:
            pred: 预测值
            target: 目标值
        
        Returns:
            损失值
        """
        return self.forward(pred, target)
    
    def __repr__(self) -> str:
        """返回损失函数的字符串表示。"""
        return f"{self.__class__.__name__}()"
