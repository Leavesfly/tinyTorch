"""带自动微分的基本算术运算。

该模块实现了基本算术运算：Add（加）、Sub（减）、Mul（乘）、Div（除）、Neg（负）。

作者：TinyAI Team
版本：0.1.0
"""

from typing import List
from tinytorch.tensor import Tensor
from tinytorch.autograd.function import Function


class Add(Function):
    """元素级加法：z = x + y
    
    前向传播：z = x + y
    反向传播：dL/dx = dL/dz, dL/dy = dL/dz
    """
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """加法的前向传播。"""
        return x.add(y)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """加法的反向传播。
        
        梯度均等地流向两个输入。
        处理广播：对广播维度求和。
        """
        x_shape = self.inputs[0].value.shape
        y_shape = self.inputs[1].value.shape
        
        grad_x = grad_output
        grad_y = grad_output
        
        # 处理广播：对广播维度求和
        if grad_x.shape != x_shape:
            grad_x = self._sum_to_shape(grad_x, x_shape)
        
        if grad_y.shape != y_shape:
            grad_y = self._sum_to_shape(grad_y, y_shape)
        
        return [grad_x, grad_y]
    
    @staticmethod
    def _sum_to_shape(tensor: Tensor, target_shape) -> Tensor:
        """将 tensor 求和到 target_shape（处理反向传播中的广播）。"""
        # 对额外维度求和
        ndim_diff = tensor.shape.ndim - target_shape.ndim
        for _ in range(ndim_diff):
            tensor = tensor.sum(axis=0, keepdims=False)
        
        # 对目标为 1 的维度求和
        for i in range(target_shape.ndim):
            if target_shape[i] == 1 and tensor.shape[i] > 1:
                tensor = tensor.sum(axis=i, keepdims=True)
        
        return tensor


class Sub(Function):
    """元素级减法：z = x - y
    
    前向传播：z = x - y
    反向传播：dL/dx = dL/dz, dL/dy = -dL/dz
    """
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """减法的前向传播。"""
        return x.sub(y)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """减法的反向传播。"""
        x_shape = self.inputs[0].value.shape
        y_shape = self.inputs[1].value.shape
        
        grad_x = grad_output
        grad_y = grad_output.neg()
        
        # 处理广播
        if grad_x.shape != x_shape:
            grad_x = Add._sum_to_shape(grad_x, x_shape)
        
        if grad_y.shape != y_shape:
            grad_y = Add._sum_to_shape(grad_y, y_shape)
        
        return [grad_x, grad_y]


class Mul(Function):
    """元素级乘法：z = x * y
    
    前向传播：z = x * y
    反向传播：dL/dx = dL/dz * y, dL/dy = dL/dz * x
    """
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """乘法的前向传播。"""
        self.save_for_backward(x, y)
        return x.mul(y)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """乘法的反向传播。"""
        x, y = self.get_saved_tensors()
        x_shape = self.inputs[0].value.shape
        y_shape = self.inputs[1].value.shape
        
        grad_x = grad_output.mul(y)
        grad_y = grad_output.mul(x)
        
        # 处理广播
        if grad_x.shape != x_shape:
            grad_x = Add._sum_to_shape(grad_x, x_shape)
        
        if grad_y.shape != y_shape:
            grad_y = Add._sum_to_shape(grad_y, y_shape)
        
        return [grad_x, grad_y]


class Div(Function):
    """元素级除法：z = x / y
    
    前向传播：z = x / y
    反向传播：dL/dx = dL/dz / y, dL/dy = -dL/dz * x / y^2
    """
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """除法的前向传播。"""
        self.save_for_backward(x, y)
        return x.div(y)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """除法的反向传播。"""
        x, y = self.get_saved_tensors()
        x_shape = self.inputs[0].value.shape
        y_shape = self.inputs[1].value.shape
        
        grad_x = grad_output.div(y)
        grad_y = grad_output.neg().mul(x).div(y.pow(2))
        
        # 处理广播
        if grad_x.shape != x_shape:
            grad_x = Add._sum_to_shape(grad_x, x_shape)
        
        if grad_y.shape != y_shape:
            grad_y = Add._sum_to_shape(grad_y, y_shape)
        
        return [grad_x, grad_y]


class Neg(Function):
    """取负运算：y = -x
    
    前向传播：y = -x
    反向传播：dL/dx = -dL/dy
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """取负的前向传播。"""
        return x.neg()
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """取负的逆向传播。"""
        return [grad_output.neg()]
