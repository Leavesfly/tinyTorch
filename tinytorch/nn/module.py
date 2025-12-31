"""神经网络模块基类。

Module 是所有神经网络层和模型的基类，提供参数管理、子模块管理、
训练/评估模式切换等核心功能。

Author: TinyAI Team
"""

from typing import Dict, List, Iterator, Tuple, Optional, Any
from tinytorch.autograd.variable import Variable


class Module:
    """神经网络模块基类。
    
    所有神经网络层和模型都应该继承这个类。子类需要实现 forward 方法
    定义前向传播逻辑。
    
    Attributes:
        _modules: 子模块字典，存储嵌套的模块
        _parameters: 参数字典，存储可训练参数
        _buffers: 缓冲区字典，存储不可训练的张量
        training: 训练/评估模式标志
        name: 模块名称
    
    Example:
        >>> class MyModule(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.param = Parameter(Tensor.randn((10, 5)))
        ...     def forward(self, x):
        ...         return x @ self.param
    """
    
    def __init__(self, name: str = None):
        """初始化模块。
        
        Args:
            name: 模块名称，如果为None则使用类名
        """
        self._modules: Dict[str, 'Module'] = {}
        self._parameters: Dict[str, 'Parameter'] = {}
        self._buffers: Dict[str, Variable] = {}
        self.training: bool = True
        self.name: str = name if name is not None else self.__class__.__name__
    
    def forward(self, *inputs) -> Variable:
        """前向传播（抽象方法，子类必须实现）。
        
        Args:
            *inputs: 输入变量
            
        Returns:
            输出变量
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward method")
    
    def __call__(self, *inputs) -> Variable:
        """调用模块，执行前向传播。
        
        Args:
            *inputs: 输入变量
            
        Returns:
            前向传播的输出
        """
        return self.forward(*inputs)
    
    def register_module(self, name: str, module: Optional['Module']) -> None:
        """注册子模块。
        
        Args:
            name: 子模块的名称
            module: 子模块实例，如果为None则删除已有的子模块
        """
        if module is None:
            if name in self._modules:
                del self._modules[name]
        else:
            if not isinstance(module, Module):
                raise TypeError(f"module must be an instance of Module, got {type(module)}")
            self._modules[name] = module
    
    def register_parameter(self, name: str, param: Optional['Parameter']) -> None:
        """注册参数。
        
        Args:
            name: 参数的名称
            param: 参数实例，如果为None则删除已有的参数
        """
        if param is None:
            if name in self._parameters:
                del self._parameters[name]
        else:
            # 延迟导入避免循环依赖
            from tinytorch.nn.parameter import Parameter
            if not isinstance(param, Parameter):
                raise TypeError(f"param must be an instance of Parameter, got {type(param)}")
            self._parameters[name] = param
    
    def register_buffer(self, name: str, tensor: Optional[Variable]) -> None:
        """注册缓冲区（不可训练的张量）。
        
        Args:
            name: 缓冲区的名称
            tensor: 张量变量，如果为None则删除已有的缓冲区
        """
        if tensor is None:
            if name in self._buffers:
                del self._buffers[name]
        else:
            if not isinstance(tensor, Variable):
                raise TypeError(f"tensor must be an instance of Variable, got {type(tensor)}")
            self._buffers[name] = tensor
    
    def parameters(self, recursive: bool = True) -> List['Parameter']:
        """获取所有参数列表。
        
        Args:
            recursive: 是否递归获取子模块的参数
            
        Returns:
            参数列表
        """
        params = []
        # 添加当前模块的参数
        params.extend(self._parameters.values())
        
        # 递归添加子模块的参数
        if recursive:
            for module in self._modules.values():
                params.extend(module.parameters(recursive=True))
        
        return params
    
    def named_parameters(self, prefix: str = '', recursive: bool = True) -> Iterator[Tuple[str, 'Parameter']]:
        """获取所有参数的名称和值。
        
        Args:
            prefix: 参数名称的前缀
            recursive: 是否递归获取子模块的参数
            
        Yields:
            (参数名称, 参数值) 的元组
        """
        # 生成当前模块的参数
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield (full_name, param)
        
        # 递归生成子模块的参数
        if recursive:
            for module_name, module in self._modules.items():
                module_prefix = f"{prefix}.{module_name}" if prefix else module_name
                yield from module.named_parameters(prefix=module_prefix, recursive=True)
    
    def modules(self) -> Iterator['Module']:
        """递归获取所有模块（包括自身）。
        
        Yields:
            模块实例
        """
        yield self
        for module in self._modules.values():
            yield from module.modules()
    
    def named_modules(self, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
        """递归获取所有模块的名称和实例。
        
        Args:
            prefix: 模块名称的前缀
            
        Yields:
            (模块名称, 模块实例) 的元组
        """
        yield (prefix, self)
        for name, module in self._modules.items():
            module_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(prefix=module_prefix)
    
    def train(self, mode: bool = True) -> 'Module':
        """设置训练模式。
        
        影响某些层的行为，如 Dropout 和 BatchNorm。
        
        Args:
            mode: True为训练模式，False为评估模式
            
        Returns:
            self，支持链式调用
        """
        self.training = mode
        # 递归设置所有子模块的模式
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self) -> 'Module':
        """设置评估模式。
        
        等价于 train(False)。
        
        Returns:
            self，支持链式调用
        """
        return self.train(False)
    
    def zero_grad(self) -> None:
        """清除所有参数的梯度。"""
        for param in self.parameters():
            param.clear_grad()
    
    def __setattr__(self, name: str, value: Any) -> None:
        """设置属性，自动注册模块和参数。
        
        Args:
            name: 属性名称
            value: 属性值
        """
        # 延迟导入避免循环依赖
        from tinytorch.nn.parameter import Parameter
        
        # 如果是特殊属性（以下划线开头），直接设置
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        # 如果是 Module 实例，注册为子模块
        elif isinstance(value, Module):
            self.register_module(name, value)
            object.__setattr__(self, name, value)
        # 如果是 Parameter 实例，注册为参数
        elif isinstance(value, Parameter):
            self.register_parameter(name, value)
            object.__setattr__(self, name, value)
        # 如果是 Variable 实例，可能是缓冲区
        elif isinstance(value, Variable) and hasattr(self, '_buffers'):
            self.register_buffer(name, value)
            object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)
    
    def __repr__(self) -> str:
        """返回模块的字符串表示。"""
        # 构建子模块列表
        lines = []
        for key, module in self._modules.items():
            # 获取子模块的表示并缩进
            mod_str = repr(module)
            mod_str = '\n'.join('  ' + line for line in mod_str.split('\n'))
            lines.append(f"({key}): {mod_str}")
        
        # 构建最终表示
        main_str = f"{self.__class__.__name__}("
        if lines:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        
        return main_str
    
    def to_dict(self) -> Dict[str, Any]:
        """将模块转换为字典（用于序列化）。
        
        Returns:
            包含模块信息的字典
        """
        state = {
            'class': self.__class__.__name__,
            'name': self.name,
            'training': self.training,
            'parameters': {},
            'buffers': {},
            'modules': {}
        }
        
        # 保存参数
        for name, param in self._parameters.items():
            state['parameters'][name] = param.to_dict()
        
        # 保存缓冲区
        for name, buffer in self._buffers.items():
            state['buffers'][name] = {
                'value': buffer.value.to_list() if hasattr(buffer.value, 'to_list') else buffer.value,
                'shape': buffer.value.shape.dims if hasattr(buffer.value, 'shape') else None
            }
        
        # 保存子模块
        for name, module in self._modules.items():
            state['modules'][name] = module.to_dict()
        
        return state
