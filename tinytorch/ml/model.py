"""模型管理类。

提供模型的生命周期管理功能，包括保存、加载、序列化等。

Author: TinyAI Team
"""

import pickle
from typing import Optional, Dict, Any, List
from tinytorch.nn.module import Module
from tinytorch.nn.parameter import Parameter
from tinytorch.autograd.tensor import Tensor


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
    
    def forward(self, input: Tensor) -> Tensor:
        """前向传播。
        
        Args:
            input: 输入变量
        
        Returns:
            输出变量
        """
        return self.module(input)
    
    def __call__(self, input: Tensor) -> Tensor:
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
    
    def train(self) -> 'Model':
        """设置为训练模式。"""
        self.module.train()
        return self
    
    def eval(self) -> 'Model':
        """设置为评估模式。"""
        self.module.eval()
        return self
    
    def zero_grad(self) -> None:
        """清除所有参数的梯度。"""
        self.module.zero_grad()
    
    def save(self, file_path: str) -> None:
        """保存模型到文件。
        
        保存模型元数据和模块状态字典。
        
        Args:
            file_path: 保存路径
        """
        model_state = {
            'name': self.name,
            'model_info': self.model_info,
            'training': self.module.training,
            'module_state_dict': self.module.state_dict(),
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_state, f)
    
    @staticmethod
    def load(file_path: str, module: Optional[Module] = None) -> 'Model':
        """从文件加载模型。
        
        Args:
            file_path: 模型文件路径
            module: 已构建好的模型模块；如果提供，则会加载保存的状态
        
        Returns:
            Model 实例
        """
        with open(file_path, 'rb') as f:
            model_state = pickle.load(f)

        if module is None:
            raise ValueError(
                "Model.load() requires a pre-constructed module instance. "
                "Build the model architecture first, then pass it via module=..."
            )

        module.load_state_dict(model_state['module_state_dict'])
        module.train(model_state.get('training', True))

        model = Model(model_state['name'], module)
        model.model_info = model_state['model_info']
        return model
    
    def save_parameters(self, file_path: str) -> None:
        """仅保存模型参数。
        
        Args:
            file_path: 保存路径
        """
        params_state = {
            'name': self.name,
            'parameters': self.module.state_dict()
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

        self.module.load_state_dict(params_state['parameters'], strict=False)
    
    def __repr__(self) -> str:
        """返回模型的字符串表示。"""
        return f"Model(name='{self.name}', module={repr(self.module)})"
