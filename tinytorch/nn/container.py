"""神经网络容器模块。

提供顺序容器等，用于组合多个神经网络层。

Author: TinyAI Team
"""

from typing import List, Optional, Iterator
from tinytorch.nn.module import Module
from tinytorch.autograd.tensor import Tensor


class Sequential(Module):
    """顺序容器。
    
    Sequential 是一个有序的模块容器。数据会按照模块添加的顺序依次
    通过每个模块进行前向传播。
    
    Attributes:
        _layers: 存储层的列表
    
    Example:
        >>> model = Sequential(
        ...     Linear(10, 20),
        ...     ReLU(),
        ...     Linear(20, 5)
        ... )
        >>> x = Tensor(NdArray.randn((32, 10)))
        >>> y = model(x)
        >>> print(y.value.shape.dims)
        (32, 5)
    """
    
    def __init__(self, *layers: Module, name: Optional[str] = None):
        """初始化顺序容器。
        
        Args:
            *layers: 要添加的层（可变参数）
            name: 容器名称
        """
        super().__init__(name=name or 'Sequential')
        self._layers = []
        
        # 添加所有传入的层
        for idx, layer in enumerate(layers):
            self.add(layer, name=f'layer_{idx}')
    
    def add(self, module: Module, name: Optional[str] = None) -> 'Sequential':
        """添加一个模块到容器末尾。
        
        Args:
            module: 要添加的模块
            name: 模块的名称，如果为 None 则自动生成
        
        Returns:
            self，支持链式调用
        """
        if not isinstance(module, Module):
            raise TypeError(f"module must be an instance of Module, got {type(module)}")
        
        # 生成模块名称（检查冲突避免覆盖已有子模块）
        if name is None:
            idx = len(self._layers)
            name = f'layer_{idx}'
            while name in self._modules:
                idx += 1
                name = f'layer_{idx}'

        # 添加到层列表
        self._layers.append(module)
        
        # 注册为子模块
        self.register_module(name, module)
        
        return self
    
    def forward(self, input: Tensor) -> Tensor:
        """前向传播。
        
        数据依次通过每个层进行前向传播。
        
        Args:
            input: 输入变量
        
        Returns:
            输出变量
        """
        output = input
        for layer in self._layers:
            output = layer(output)
        return output
    
    def __len__(self) -> int:
        """返回容器中的层数。"""
        return len(self._layers)
    
    @property
    def layers(self) -> List[Module]:
        """获取层列表（_layers 的别名）。"""
        return self._layers
    
    def __getitem__(self, idx: int) -> Module:
        """通过索引获取层。
        
        Args:
            idx: 层的索引
        
        Returns:
            对应的层
        """
        return self._layers[idx]
    
    def __iter__(self) -> Iterator[Module]:
        """返回层的迭代器。"""
        return iter(self._layers)
    
    def __repr__(self) -> str:
        """返回容器的字符串表示。"""
        children = ((str(idx), module) for idx, module in enumerate(self._layers))
        return self._format_child_modules(children, self.__class__.__name__)


class ModuleList(Module):
    """模块列表容器。
    
    ModuleList 类似于 Python 的 list，但会正确注册子模块，使其参数
    可以被正确追踪。与 Sequential 不同，ModuleList 不定义前向传播逻辑，
    需要在外部定义如何使用这些模块。
    
    Example:
        >>> layers = ModuleList([
        ...     Linear(10, 20),
        ...     Linear(20, 30)
        ... ])
        >>> # 需要在 forward 中手动定义如何使用这些层
    """
    
    def __init__(self, modules: Optional[List[Module]] = None, name: Optional[str] = None):
        """初始化模块列表。
        
        Args:
            modules: 要添加的模块列表
            name: 容器名称
        """
        super().__init__(name=name or 'ModuleList')
        self._modules_list = []
        
        if modules is not None:
            for module in modules:
                self.append(module)
    
    def append(self, module: Module) -> 'ModuleList':
        """在列表末尾添加一个模块。
        
        Args:
            module: 要添加的模块
        
        Returns:
            self，支持链式调用
        """
        if not isinstance(module, Module):
            raise TypeError(f"module must be an instance of Module, got {type(module)}")
        
        idx = len(self._modules_list)
        self._modules_list.append(module)
        self.register_module(str(idx), module)
        
        return self
    
    def forward(self, *inputs):
        """ModuleList 不实现 forward 方法。"""
        raise NotImplementedError("ModuleList does not implement forward, use it in a custom module")
    
    def __len__(self) -> int:
        """返回列表中的模块数量。"""
        return len(self._modules_list)
    
    def __getitem__(self, idx: int) -> Module:
        """通过索引获取模块。
        
        Args:
            idx: 模块的索引
        
        Returns:
            对应的模块
        """
        return self._modules_list[idx]
    
    def __iter__(self) -> Iterator[Module]:
        """返回模块的迭代器。"""
        return iter(self._modules_list)
    
    def __repr__(self) -> str:
        """返回模块列表的字符串表示。"""
        children = ((str(idx), module) for idx, module in enumerate(self._modules_list))
        return self._format_child_modules(children, self.__class__.__name__)