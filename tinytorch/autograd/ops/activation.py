"""带自动微分的激活函数运算。

该模块实现了激活函数：ReLU、Sigmoid、Tanh。

作者：TinyAI Team
版本：0.1.0
"""

import math
from typing import List
from tinytorch.tensor import Tensor
from tinytorch.autograd.function import Function


class ReLU(Function):
    """ReLU 激活函数：y = max(0, x)
    
    前向传播：y = max(0, x)
    反向传播：dL/dx = dL/dy * (x > 0)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """ReLU 的前向传播。"""
        self.save_for_backward(x)
        return x.relu()
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """ReLU 的反向传播。"""
        x, = self.get_saved_tensors()
        
        # 梯度在 x > 0 时为 1，否则为 0
        mask_data = [1.0 if val > 0 else 0.0 for val in x.data]
        mask = Tensor(mask_data, x.shape, x.dtype)
        
        grad_x = grad_output.mul(mask)
        return [grad_x]


class Sigmoid(Function):
    """Sigmoid 激活函数：y = 1 / (1 + exp(-x))
    
    前向传播：y = sigmoid(x)
    反向传播：dL/dx = dL/dy * y * (1 - y)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Sigmoid 的前向传播。"""
        y = x.sigmoid()
        self.save_for_backward(y)
        return y
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Sigmoid 的反向传播。"""
        y, = self.get_saved_tensors()
        
        # dy/dx = y * (1 - y) 的梯度计算
        one = Tensor.ones(y.shape, y.dtype)
        grad_x = grad_output.mul(y).mul(one.sub(y))
        return [grad_x]


class Tanh(Function):
    """Tanh 激活函数：y = tanh(x)
    
    前向传播：y = tanh(x)
    反向传播：dL/dx = dL/dy * (1 - y^2)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Tanh 的前向传播。"""
        y = x.tanh()
        self.save_for_backward(y)
        return y
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """Tanh 的反向传播。"""
        y, = self.get_saved_tensors()
        
        # dy/dx = 1 - y^2 的梯度计算
        one = Tensor.ones(y.shape, y.dtype)
        grad_x = grad_output.mul(one.sub(y.pow(2)))
        return [grad_x]
