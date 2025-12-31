"""激活函数层封装。

将 autograd.ops.activation 中的激活函数封装为神经网络层。

Author: TinyAI Team
"""

from tinytorch.nn.module import Module
from tinytorch.autograd.variable import Variable


class ReLU(Module):
    """ReLU 激活层。
    
    ReLU(x) = max(0, x)
    
    Example:
        >>> relu = ReLU()
        >>> x = Variable(Tensor([[-1, 2], [3, -4]]))
        >>> y = relu(x)
        >>> print(y.value.to_list())
        [[0, 2], [3, 0]]
    """
    
    def __init__(self, name: str = None):
        """初始化 ReLU 层。
        
        Args:
            name: 层的名称
        """
        super().__init__(name=name or 'ReLU')
    
    def forward(self, input: Variable) -> Variable:
        """前向传播。
        
        Args:
            input: 输入变量
        
        Returns:
            输出变量
        """
        return input.relu()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Sigmoid(Module):
    """Sigmoid 激活层。
    
    Sigmoid(x) = 1 / (1 + exp(-x))
    
    Example:
        >>> sigmoid = Sigmoid()
        >>> x = Variable(Tensor([[0, 1], [2, -1]]))
        >>> y = sigmoid(x)
    """
    
    def __init__(self, name: str = None):
        """初始化 Sigmoid 层。
        
        Args:
            name: 层的名称
        """
        super().__init__(name=name or 'Sigmoid')
    
    def forward(self, input: Variable) -> Variable:
        """前向传播。
        
        Args:
            input: 输入变量
        
        Returns:
            输出变量
        """
        return input.sigmoid()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Tanh(Module):
    """Tanh 激活层。
    
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Example:
        >>> tanh = Tanh()
        >>> x = Variable(Tensor([[0, 1], [2, -1]]))
        >>> y = tanh(x)
    """
    
    def __init__(self, name: str = None):
        """初始化 Tanh 层。
        
        Args:
            name: 层的名称
        """
        super().__init__(name=name or 'Tanh')
    
    def forward(self, input: Variable) -> Variable:
        """前向传播。
        
        Args:
            input: 输入变量
        
        Returns:
            输出变量
        """
        return input.tanh()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class LeakyReLU(Module):
    """LeakyReLU 激活层。
    
    LeakyReLU(x) = max(negative_slope * x, x)
    
    Example:
        >>> leaky_relu = LeakyReLU(negative_slope=0.01)
        >>> x = Variable(Tensor([[-1, 2], [3, -4]]))
        >>> y = leaky_relu(x)
    """
    
    def __init__(self, negative_slope: float = 0.01, name: str = None):
        """初始化 LeakyReLU 层。
        
        Args:
            negative_slope: 负半轴的斜率，默认为 0.01
            name: 层的名称
        """
        super().__init__(name=name or 'LeakyReLU')
        self.negative_slope = negative_slope
    
    def forward(self, input: Variable) -> Variable:
        """前向传播。
        
        Args:
            input: 输入变量
        
        Returns:
            输出变量
        """
        return input.leaky_relu(self.negative_slope)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(negative_slope={self.negative_slope})"
