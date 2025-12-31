"""优化器基类。

Author: TinyAI Team
"""

from typing import List, Dict, Any
from tinytorch.nn.parameter import Parameter


class Optimizer:
    """优化器基类。
    
    所有优化器都应该继承这个类。子类需要实现 step 方法定义参数更新逻辑。
    
    Attributes:
        params: 要优化的参数列表
        learning_rate: 学习率
        state: 优化器状态字典，存储动量等信息
    
    Example:
        >>> optimizer = SGD(model.parameters(), learning_rate=0.01)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """
    
    def __init__(self, params: List[Parameter], learning_rate: float = 0.01):
        """初始化优化器。
        
        Args:
            params: 要优化的参数列表
            learning_rate: 学习率
        """
        self.params = params
        self.learning_rate = learning_rate
        self.state = {}  # 用于存储优化器状态（如动量）
    
    def step(self) -> None:
        """执行一步参数更新（抽象方法，子类必须实现）。
        
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement step method")
    
    def zero_grad(self) -> None:
        """清除所有参数的梯度。"""
        for param in self.params:
            param.clear_grad()
    
    def state_dict(self) -> Dict[str, Any]:
        """获取优化器状态。
        
        Returns:
            包含优化器状态的字典
        """
        return {
            'learning_rate': self.learning_rate,
            'state': self.state
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载优化器状态。
        
        Args:
            state_dict: 包含优化器状态的字典
        """
        self.learning_rate = state_dict.get('learning_rate', self.learning_rate)
        self.state = state_dict.get('state', {})
