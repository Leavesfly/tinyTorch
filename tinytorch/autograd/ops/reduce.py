"""带自动微分的归约运算。

该模块实现了归约运算：Sum（求和）、Mean（平均）。

作者：TinyAI Team
版本：0.1.0
"""

from typing import List, Optional
from tinytorch.tensor import Tensor
from tinytorch.autograd.function import Function


class Sum(Function):
    """求和归约：y = sum(x, axis, keepdims)
    
    前向传播：y = sum(x, axis, keepdims)
    反向传播：dL/dx = broadcast(dL/dy) to x.shape
    """
    
    def __init__(self, axis: Optional[int] = None, keepdims: bool = False):
        """初始化求和归约。
        
        Args:
            axis: 求和的轴（None 表示所有维度）
            keepdims: 是否保留归约后的维度
        """
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
    
    def forward(self, x: Tensor) -> Tensor:
        """求和的前向传播。"""
        self.input_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """求和的反向传播。"""
        # 将梯度广播回输入形状
        if self.keepdims:
            # 广播后形状已经匹配
            grad_x = grad_output._broadcast_to(self.input_shape)
        else:
            # 需要重新添加归约的维度
            if self.axis is None:
                # 对所有维度求和
                grad_x = Tensor([grad_output.data[0]] * self.input_shape.size, 
                              self.input_shape, grad_output.dtype)
            else:
                # 对特定轴求和
                # 扩展维度并广播
                new_shape_list = list(grad_output.shape.dims)
                axis = self.axis if self.axis >= 0 else self.input_shape.ndim + self.axis
                new_shape_list.insert(axis, 1)
                grad_expanded = grad_output.reshape(tuple(new_shape_list))
                grad_x = grad_expanded._broadcast_to(self.input_shape)
        
        return [grad_x]


class Mean(Function):
    """平均归约：y = mean(x, axis, keepdims)
    
    前向传播：y = mean(x, axis, keepdims)
    反向传播：dL/dx = broadcast(dL/dy / count) to x.shape
    """
    
    def __init__(self, axis: Optional[int] = None, keepdims: bool = False):
        """初始化平均归约。
        
        Args:
            axis: 求平均的轴（None 表示所有维度）
            keepdims: 是否保留归约后的维度
        """
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
    
    def forward(self, x: Tensor) -> Tensor:
        """平均的前向传播。"""
        self.input_shape = x.shape
        
        # 为反向传播计算计数
        if self.axis is None:
            self.count = x.shape.size
        else:
            axis = self.axis if self.axis >= 0 else x.shape.ndim + self.axis
            self.count = x.shape.dims[axis]
        
        return x.mean(axis=self.axis, keepdims=self.keepdims)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """平均的反向传播。"""
        # 梯度除以计数
        grad_output = grad_output.div(self.count)
        
        # 将梯度广播回输入形状（与 Sum 相同）
        if self.keepdims:
            grad_x = grad_output._broadcast_to(self.input_shape)
        else:
            if self.axis is None:
                grad_x = Tensor([grad_output.data[0]] * self.input_shape.size,
                              self.input_shape, grad_output.dtype)
            else:
                new_shape_list = list(grad_output.shape.dims)
                axis = self.axis if self.axis >= 0 else self.input_shape.ndim + self.axis
                new_shape_list.insert(axis, 1)
                grad_expanded = grad_output.reshape(tuple(new_shape_list))
                grad_x = grad_expanded._broadcast_to(self.input_shape)
        
        return [grad_x]
