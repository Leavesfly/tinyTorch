"""自动微分操作的函数基类。

本模块提供了所有可微分操作必须继承的 Function 基类。
它定义了前向传播和反向传播的接口。

Author: TinyAI Team
Version: 0.1.0
"""

from typing import List, Tuple
from tinytorch.tensor import Tensor


class Function:
    """所有可微分函数的基类。
    
    子类必须实现 forward() 和 backward() 方法。
    Function 维护对输入变量和保存张量的引用，以便进行反向传播。
    
    属性:
        inputs: 输入 Variable 对象列表
        outputs: 输出 Variable 对象列表
        saved_tensors: 前向传播时保存的张量列表
    
    示例:
        >>> class Add(Function):
        ...     def forward(self, x, y):
        ...         return x.add(y)
        ...     def backward(self, grad_output):
        ...         return [grad_output, grad_output]
    """
    
    def __init__(self):
        """初始化 Function。"""
        self.inputs = []
        self.outputs = []
        self.saved_tensors = []
    
    def __call__(self, *inputs):
        """调用函数（与 call 方法相同）。
        
        参数:
            *inputs: Variable 输入
            
        返回:
            输出 Variable
        """
        return self.call(*inputs)
    
    def call(self, *inputs):
        """执行前向传播并构建计算图。
        
        此方法:
        1. 从输入 Variables 中提取张量值
        2. 调用 forward() 计算输出张量
        3. 将输出张量包装成 Variables
        4. 设置反向传播图连接
        
        参数:
            *inputs: Variable 输入
            
        返回:
            输出 Variable（或多输出的 Variables 元组）
        """
        # 在此处导入以避免循环依赖
        from tinytorch.autograd.variable import Variable
        
        # 存储输入变量
        self.inputs = list(inputs)
        
        # 提取张量值
        input_tensors = [var.value for var in inputs]
        
        # 前向传播
        output_tensors = self.forward(*input_tensors)
        
        # 处理单输出或多输出
        if isinstance(output_tensors, (list, tuple)):
            # 多输出
            output_vars = []
            for tensor in output_tensors:
                var = Variable(tensor)
                var.creator = self
                output_vars.append(var)
            self.outputs = output_vars
            return tuple(output_vars)
        else:
            # 单输出
            output_var = Variable(output_tensors)
            output_var.creator = self
            self.outputs = [output_var]
            return output_var
    
    def forward(self, *inputs: Tensor) -> Tensor:
        """前向传播计算。
        
        子类必须实现此方法以定义前向计算。
        
        参数:
            *inputs: 输入张量
            
        返回:
            输出张量
            
        异常:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        """反向传播计算。
        
        子类必须实现此方法，根据输出梯度计算输入梯度。
        
        参数:
            grad_output: 损失对输出的梯度
            
        返回:
            每个输入的梯度列表
            
        异常:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement backward()")
    
    def save_for_backward(self, *tensors: Tensor):
        """保存反向计算所需的张量。
        
        参数:
            *tensors: 要保存的张量
        """
        self.saved_tensors = list(tensors)
    
    def get_saved_tensors(self) -> Tuple[Tensor, ...]:
        """获取保存的张量。
        
        返回:
            保存的张量元组
        """
        return tuple(self.saved_tensors)
