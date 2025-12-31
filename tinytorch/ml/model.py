"""模型管理类。

提供模型的生命周期管理功能，包括保存、加载、序列化等。

Author: TinyAI Team
"""

import pickle
from typing import Optional, Dict, Any, List
from tinytorch.nn.module import Module
from tinytorch.nn.parameter import Parameter
from tinytorch.autograd.variable import Variable


class Model:
    """模型管理类。
    
    封装神经网络模块，提供模型保存、加载、参数管理等功能。
    
    Attributes:
        name: 模型名称
        module: 神经网络模块
        model_info: 模型元数据信息
    
    Example:
        >>> from tinytorch.nn import Sequential, Linear, ReLU
        >>> net = Sequential(
        ...     Linear(10, 20),
        ...     ReLU(),
        ...     Linear(20, 5)
        ... )
        >>> model = Model(name='MyModel', module=net)
        >>> model.save('model.pkl')
        >>> loaded_model = Model.load('model.pkl')
    """
    
    def __init__(self, name: str, module: Module):
        """初始化模型。
        
        Args:
            name: 模型名称
            module: 神经网络模块
        """
        self.name = name
        self.module = module
        self.model_info = {
            'name': name,
            'version': '1.0',
            'framework': 'tinyTorch'
        }
    
    def forward(self, input: Variable) -> Variable:
        """前向传播。
        
        Args:
            input: 输入变量
        
        Returns:
            输出变量
        """
        return self.module(input)
    
    def __call__(self, input: Variable) -> Variable:
        """调用模型，执行前向传播。
        
        Args:
            input: 输入变量
        
        Returns:
            输出变量
        """
        return self.forward(input)
    
    def parameters(self) -> List[Parameter]:
        """获取所有参数。
        
        Returns:
            参数列表
        """
        return self.module.parameters()
    
    def named_parameters(self):
        """获取所有参数的名称和值。
        
        Returns:
            (参数名称, 参数值) 的迭代器
        """
        return self.module.named_parameters()
    
    def train(self) -> None:
        """设置为训练模式。"""
        self.module.train()
    
    def eval(self) -> None:
        """设置为评估模式。"""
        self.module.eval()
    
    def zero_grad(self) -> None:
        """清除所有参数的梯度。"""
        self.module.zero_grad()
    
    def save(self, file_path: str) -> None:
        """保存模型到文件。
        
        使用 pickle 序列化整个模型（包括结构和参数）。
        
        Args:
            file_path: 保存路径
        """
        model_state = {
            'name': self.name,
            'model_info': self.model_info,
            'module_state': self.module.to_dict()
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_state, f)
    
    @staticmethod
    def load(file_path: str) -> 'Model':
        """从文件加载模型。
        
        Args:
            file_path: 模型文件路径
        
        Returns:
            Model 实例
        """
        with open(file_path, 'rb') as f:
            model_state = pickle.load(f)
        
        # 注意：这里简化处理，实际使用时需要先构建相同结构的模型，
        # 然后加载参数。完整实现需要模型注册机制。
        print(f"Warning: Model.load() is simplified. "
              f"Please construct model manually and use load_parameters() instead.")
        
        # 返回一个占位模型
        from tinytorch.nn import Sequential
        dummy_module = Sequential()
        model = Model(model_state['name'], dummy_module)
        model.model_info = model_state['model_info']
        
        return model
    
    def save_parameters(self, file_path: str) -> None:
        """仅保存模型参数。
        
        Args:
            file_path: 保存路径
        """
        params_dict = {}
        for name, param in self.named_parameters():
            params_dict[name] = param.to_dict()
        
        params_state = {
            'name': self.name,
            'parameters': params_dict
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(params_state, f)
    
    def load_parameters(self, file_path: str) -> None:
        """加载模型参数。
        
        Args:
            file_path: 参数文件路径
        """
        with open(file_path, 'rb') as f:
            params_state = pickle.load(f)
        
        # 加载参数到模型
        saved_params = params_state['parameters']
        for name, param in self.named_parameters():
            if name in saved_params:
                param_data = saved_params[name]
                # 更新参数值
                from tinytorch.tensor.tensor import Tensor
                param.value = Tensor(param_data['value'])
    
    def __repr__(self) -> str:
        """返回模型的字符串表示。"""
        return f"Model(name='{self.name}', module={repr(self.module)})"
