"""全连接层（线性层）实现。

Author: TinyAI Team
"""

from typing import Optional
from tinytorch.nn.module import Module
from tinytorch.nn.parameter import Parameter
from tinytorch.autograd.variable import Variable
from tinytorch.tensor.tensor import Tensor
from tinytorch.nn import init


class Linear(Module):
    """全连接层（线性层）。
    
    实现线性变换: y = xW^T + b
    
    其中 x 的形状为 (batch_size, in_features)
         W 的形状为 (out_features, in_features)
         b 的形状为 (out_features,)
         y 的形状为 (batch_size, out_features)
    
    Attributes:
        in_features: 输入特征数
        out_features: 输出特征数
        weight: 权重参数，形状为 (out_features, in_features)
        bias: 偏置参数，形状为 (out_features,)，如果 use_bias=False 则为 None
    
    Example:
        >>> layer = Linear(10, 5)
        >>> x = Variable(Tensor.randn((32, 10)))
        >>> y = layer(x)
        >>> print(y.value.shape.dims)
        (32, 5)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 use_bias: bool = True, name: str = None):
        """初始化全连接层。
        
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            use_bias: 是否使用偏置，默认为 True
            name: 层的名称
        """
        super().__init__(name=name or 'Linear')
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        
        # 初始化权重参数 (out_features, in_features)
        weight_tensor = Tensor.zeros((out_features, in_features))
        self.weight = Parameter(weight_tensor, name=f'{self.name}.weight')
        
        # 初始化偏置参数 (out_features,)
        if use_bias:
            bias_tensor = Tensor.zeros((out_features,))
            self.bias = Parameter(bias_tensor, name=f'{self.name}.bias')
        else:
            self.bias = None
        
        # 使用 Kaiming 初始化（He 初始化，适合 ReLU）
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """重置参数。
        
        使用 Kaiming 均匀分布初始化权重，偏置初始化为零。
        """
        # Kaiming 均匀分布初始化权重
        init.kaiming_uniform_(self.weight.value, nonlinearity='relu')
        
        # 偏置初始化为零
        if self.bias is not None:
            init.zeros_(self.bias.value)
    
    def forward(self, input: Variable) -> Variable:
        """前向传播。
        
        计算: output = input @ weight^T + bias
        
        Args:
            input: 输入变量，形状为 (batch_size, in_features)
        
        Returns:
            输出变量，形状为 (batch_size, out_features)
        
        Raises:
            ValueError: 当输入形状不匹配时
        """
        # 检查输入形状
        if len(input.value.shape.dims) != 2:
            raise ValueError(
                f"Linear layer expects 2D input, got {len(input.value.shape.dims)}D"
            )
        
        if input.value.shape.dims[1] != self.in_features:
            raise ValueError(
                f"Linear layer expects input with {self.in_features} features, "
                f"got {input.value.shape.dims[1]}"
            )
        
        # 计算 output = input @ weight^T
        # weight 的形状是 (out_features, in_features)
        # 需要转置为 (in_features, out_features)
        weight_transposed = self.weight.transpose()
        output = input.matmul(weight_transposed)
        
        # 加上偏置
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def __repr__(self) -> str:
        """返回层的字符串表示。"""
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"out_features={self.out_features}, use_bias={self.use_bias})")
