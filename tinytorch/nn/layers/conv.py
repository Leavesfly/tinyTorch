"""卷积层。

Author: TinyAI Team
"""

from tinytorch.nn.module import Module
from tinytorch.nn.parameter import Parameter
from tinytorch.autograd import Tensor
from tinytorch.autograd.ops.conv import Conv2d as _Conv2dOp
from tinytorch.ndarr import NdArray
from tinytorch.nn import init


class Conv2d(Module):
    """二维卷积层。
    
    对输入进行二维卷积操作。输入形状为 (batch_size, in_channels, height, width)。
    
    Attributes:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        use_bias: 是否使用偏置
        weight: 卷积核权重参数
        bias: 偏置参数
    
    Example:
        >>> conv = Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        >>> x = Tensor(NdArray.randn((1, 3, 32, 32)))
        >>> y = conv(x)
        >>> print(y.value.shape)
        (1, 64, 32, 32)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, use_bias: bool = True):
        """初始化 Conv2d 层。
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小（正方形）
            stride: 步长
            padding: 填充大小
            use_bias: 是否使用偏置
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        
        # 初始化权重：(out_channels, in_channels, kernel_size, kernel_size)
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        weight_tensor = NdArray.zeros(weight_shape)
        init.kaiming_uniform_(weight_tensor)
        self.weight = Parameter(weight_tensor, name='weight')
        
        # 初始化偏置：(out_channels,)
        if use_bias:
            bias_tensor = NdArray.zeros((out_channels,))
            self.bias = Parameter(bias_tensor, name='bias')
        else:
            self.bias = None
    
    def forward(self, input: Tensor) -> Tensor:
        """前向传播。
        
        Args:
            input: 输入张量，形状 (batch_size, in_channels, height, width)
        
        Returns:
            输出张量，形状 (batch_size, out_channels, out_height, out_width)
        """
        batch_size, in_channels, height, width = input.value.shape.dims
        
        if in_channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {in_channels}"
            )
        
        op = _Conv2dOp(
            stride=self.stride,
            padding=self.padding,
            kernel_size=self.kernel_size,
            use_bias=self.use_bias,
        )
        if self.use_bias and self.bias is not None:
            return op(input, self.weight, self.bias)
        return op(input, self.weight)
    
    def __repr__(self) -> str:
        """返回层的字符串表示。"""
        return (f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")
