"""卷积层。

Author: TinyAI Team
"""

from tinytorch.nn.module import Module
from tinytorch.nn.parameter import Parameter
from tinytorch.autograd import Variable
from tinytorch.tensor import Tensor, Shape
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
        >>> x = Variable(Tensor.randn((1, 3, 32, 32)))
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
        weight_tensor = init.kaiming_uniform(weight_shape)
        self.weight = Parameter(weight_tensor, name='weight')
        
        # 初始化偏置：(out_channels,)
        if use_bias:
            bias_tensor = Tensor.zeros((out_channels,))
            self.bias = Parameter(bias_tensor, name='bias')
        else:
            self.bias = None
    
    def forward(self, input: Variable) -> Variable:
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
        
        # 计算输出尺寸
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 对输入进行填充
        if self.padding > 0:
            padded_input = self._pad_input(input.value, self.padding)
        else:
            padded_input = input.value
        
        # 执行卷积
        output_data = []
        
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # 计算感受野的起始位置
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        
                        # 卷积操作：对所有输入通道求和
                        conv_sum = 0.0
                        for ic in range(self.in_channels):
                            for kh in range(self.kernel_size):
                                for kw in range(self.kernel_size):
                                    # 输入索引
                                    h_idx = h_start + kh
                                    w_idx = w_start + kw
                                    
                                    # 获取输入值
                                    input_idx = (b * in_channels * padded_input.shape.dims[2] * padded_input.shape.dims[3] +
                                                ic * padded_input.shape.dims[2] * padded_input.shape.dims[3] +
                                                h_idx * padded_input.shape.dims[3] +
                                                w_idx)
                                    input_val = padded_input.data[input_idx]
                                    
                                    # 获取权重值
                                    weight_idx = (oc * self.in_channels * self.kernel_size * self.kernel_size +
                                                 ic * self.kernel_size * self.kernel_size +
                                                 kh * self.kernel_size +
                                                 kw)
                                    weight_val = self.weight.value.data[weight_idx]
                                    
                                    conv_sum += input_val * weight_val
                        
                        # 加上偏置
                        if self.use_bias:
                            conv_sum += self.bias.value.data[oc]
                        
                        output_data.append(conv_sum)
        
        # 构造输出张量
        output_shape = Shape((batch_size, self.out_channels, out_height, out_width))
        output_tensor = Tensor(output_data, output_shape, 'float32')
        
        return Variable(output_tensor, requires_grad=input.requires_grad)
    
    def _pad_input(self, input_tensor: Tensor, padding: int) -> Tensor:
        """对输入进行零填充。
        
        Args:
            input_tensor: 输入张量，形状 (batch_size, channels, height, width)
            padding: 填充大小
        
        Returns:
            填充后的张量
        """
        batch_size, channels, height, width = input_tensor.shape.dims
        padded_height = height + 2 * padding
        padded_width = width + 2 * padding
        
        # 创建填充后的张量（全零）
        padded_data = [0.0] * (batch_size * channels * padded_height * padded_width)
        
        # 将原始数据复制到填充后的张量中
        for b in range(batch_size):
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        # 原始索引
                        src_idx = (b * channels * height * width +
                                  c * height * width +
                                  h * width +
                                  w)
                        
                        # 填充后的索引
                        dst_idx = (b * channels * padded_height * padded_width +
                                  c * padded_height * padded_width +
                                  (h + padding) * padded_width +
                                  (w + padding))
                        
                        padded_data[dst_idx] = input_tensor.data[src_idx]
        
        padded_shape = Shape((batch_size, channels, padded_height, padded_width))
        return Tensor(padded_data, padded_shape, input_tensor.dtype)
    
    def __repr__(self) -> str:
        """返回层的字符串表示。"""
        return (f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")
