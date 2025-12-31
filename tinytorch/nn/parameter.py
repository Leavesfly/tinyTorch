"""神经网络参数类。

Parameter 是 Variable 的子类，专门用于神经网络的可训练参数。
与普通 Variable 的主要区别是 requires_grad 默认为 True。

Author: TinyAI Team
"""

from typing import Optional, Dict, Any
from tinytorch.tensor.tensor import Tensor
from tinytorch.autograd.variable import Variable


class Parameter(Variable):
    """可训练参数类。
    
    Parameter 是 Variable 的一个特殊子类，用于表示神经网络中的
    可训练参数（如权重和偏置）。
    
    Attributes:
        value: 参数的值（Tensor）
        grad: 参数的梯度（Tensor）
        requires_grad: 是否需要计算梯度（固定为 True）
        name: 参数名称
        creator: 创建该参数的函数（用于构建计算图）
    
    Example:
        >>> weight = Parameter(Tensor.randn((10, 5)), name='weight')
        >>> bias = Parameter(Tensor.zeros((5,)), name='bias')
        >>> output = input @ weight + bias
    """
    
    def __init__(self, value: Tensor, name: str = None):
        """初始化参数。
        
        Args:
            value: 参数的初始值（Tensor）
            name: 参数名称，用于调试和可视化
        """
        # 参数默认需要梯度
        super().__init__(value, name=name, requires_grad=True)
    
    def __repr__(self) -> str:
        """返回参数的字符串表示。
        
        Returns:
            参数的描述字符串
        """
        shape_str = str(self.value.shape.dims) if hasattr(self.value, 'shape') else 'unknown'
        name_str = f"'{self.name}'" if self.name else 'unnamed'
        return f"Parameter({name_str}, shape={shape_str}, requires_grad={self.requires_grad})"
    
    def to_dict(self) -> Dict[str, Any]:
        """将参数转换为字典（用于序列化）。
        
        Returns:
            包含参数信息的字典
        """
        return {
            'name': self.name,
            'value': self.value.to_list() if hasattr(self.value, 'to_list') else self.value,
            'shape': self.value.shape.dims if hasattr(self.value, 'shape') else None,
            'requires_grad': self.requires_grad
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Parameter':
        """从字典加载参数。
        
        Args:
            data: 包含参数信息的字典
            
        Returns:
            Parameter 实例
        """
        value = Tensor(data['value'])
        name = data.get('name')
        return Parameter(value, name=name)
