"""神经网络模块基类。

Module 是所有神经网络层和模型的基类，提供参数管理、子模块管理、
训练/评估模式切换等核心功能。

Author: TinyAI Team
"""

from typing import Dict, List, Iterator, Tuple, Optional, Any
from tinytorch.autograd.tensor import Tensor


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
        ...         self.param = Parameter(NdArray.randn((10, 5)))
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
        self._buffers: Dict[str, Tensor] = {}
        self.training: bool = True
        self.name: str = name if name is not None else self.__class__.__name__
    
    def forward(self, *inputs) -> Tensor:
        """前向传播（抽象方法，子类必须实现）。
        
        Args:
            *inputs: 输入变量
            
        Returns:
            输出变量
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward method")
    
    def __call__(self, *inputs, **kwargs) -> Tensor:
        """调用模块，执行前向传播。
        
        Args:
            *inputs: 输入变量
            **kwargs: 关键字输入参数
            
        Returns:
            前向传播的输出
        """
        return self.forward(*inputs, **kwargs)
    
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
    
    def register_buffer(self, name: str, tensor: Optional[Tensor]) -> None:
        """注册缓冲区（不可训练的张量）。
        
        Args:
            name: 缓冲区的名称
            tensor: 张量变量，如果为None则删除已有的缓冲区
        """
        if tensor is None:
            if name in self._buffers:
                del self._buffers[name]
            if name in self.__dict__:
                object.__delattr__(self, name)
        else:
            if not isinstance(tensor, Tensor):
                raise TypeError(f"ndarr must be an instance of Tensor, got {type(tensor)}")
            self._buffers[name] = tensor
            object.__setattr__(self, name, tensor)
    
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

    def named_buffers(self, prefix: str = '', recursive: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """获取所有缓冲区的名称和值。"""
        for name, buffer in self._buffers.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield (full_name, buffer)

        if recursive:
            for module_name, module in self._modules.items():
                module_prefix = f"{prefix}.{module_name}" if prefix else module_name
                yield from module.named_buffers(prefix=module_prefix, recursive=True)
    
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

        modules = self.__dict__.get('_modules')
        parameters = self.__dict__.get('_parameters')
        buffers = self.__dict__.get('_buffers')
        
        # 如果是特殊属性（以下划线开头），直接设置
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        # 如果是 Module 实例，注册为子模块
        elif isinstance(value, Module):
            if parameters is not None and name in parameters:
                self.register_parameter(name, None)
            if buffers is not None and name in buffers:
                self.register_buffer(name, None)
            self.register_module(name, value)
            object.__setattr__(self, name, value)
        # 如果是 Parameter 实例，注册为参数
        elif isinstance(value, Parameter):
            if modules is not None and name in modules:
                self.register_module(name, None)
            if buffers is not None and name in buffers:
                self.register_buffer(name, None)
            self.register_parameter(name, value)
            object.__setattr__(self, name, value)
        else:
            # 覆盖已有参数/模块/缓冲区时，保持注册表与属性一致。
            if modules is not None and name in modules:
                self.register_module(name, None)
            if parameters is not None and name in parameters:
                self.register_parameter(name, None)
            if buffers is not None and name in buffers:
                self.register_buffer(name, None)
            object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """删除属性并同步清理注册表。"""
        modules = self.__dict__.get('_modules')
        parameters = self.__dict__.get('_parameters')
        buffers = self.__dict__.get('_buffers')

        if modules is not None and name in modules:
            self.register_module(name, None)
        if parameters is not None and name in parameters:
            self.register_parameter(name, None)
        if buffers is not None and name in buffers:
            self.register_buffer(name, None)

        object.__delattr__(self, name)
    
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

    @staticmethod
    def _tensor_state(tensor: Tensor, kind: str) -> Dict[str, Any]:
        """将 Tensor 序列化为统一的状态字典。"""
        return {
            'kind': kind,
            'value': tensor.value.to_list(),
            'shape': tensor.value.shape.dims,
            'dtype': tensor.value.dtype,
            'requires_grad': tensor.requires_grad,
        }

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        """返回扁平化的参数/缓冲区状态。"""
        state = {}
        for name, param in self.named_parameters():
            state[name] = self._tensor_state(param, kind='parameter')
        for name, buffer in self.named_buffers():
            state[name] = self._tensor_state(buffer, kind='buffer')
        return state

    def load_state_dict(self, state_dict: Dict[str, Dict[str, Any]], strict: bool = True) -> Dict[str, List[str]]:
        """从扁平化状态字典加载参数和缓冲区。"""
        from tinytorch.ndarr.ndarray import NdArray
        from tinytorch.ndarr.shape import Shape

        missing_keys = []
        unexpected_keys = []

        expected_parameters = dict(self.named_parameters())
        expected_buffers = dict(self.named_buffers())
        expected_keys = set(expected_parameters) | set(expected_buffers)

        def _load_tensor(entry: Dict[str, Any], existing_value) -> NdArray:
            shape_dims = entry.get('shape')
            shape = Shape(tuple(shape_dims)) if shape_dims is not None else existing_value.shape
            return NdArray(entry['value'], shape, dtype=entry.get('dtype', existing_value.dtype))

        for name, param in expected_parameters.items():
            if name not in state_dict:
                missing_keys.append(name)
                continue
            entry = state_dict[name]
            param.value = _load_tensor(entry, param.value)

        for name, buffer in expected_buffers.items():
            if name not in state_dict:
                missing_keys.append(name)
                continue
            entry = state_dict[name]
            buffer.value = _load_tensor(entry, buffer.value)

        for key in state_dict:
            if key not in expected_keys:
                unexpected_keys.append(key)

        if strict and (missing_keys or unexpected_keys):
            raise KeyError(
                f"Error(s) in loading state_dict: missing_keys={missing_keys}, "
                f"unexpected_keys={unexpected_keys}"
            )

        return {
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """将模块转换为字典（用于序列化）。
        
        Returns:
            包含模块信息的字典
        """
        state = {
            'class': self.__class__.__name__,
            'name': self.name,
            'training': self.training,
            'state_dict': self.state_dict(),
        }
        
        return state
