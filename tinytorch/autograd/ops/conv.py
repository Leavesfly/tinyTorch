"""卷积运算 - 支持自动微分。

本模块实现二维卷积运算 Conv2d，用于构建卷积神经网络。

卷积运算将卷积核在输入上滑动，计算局部特征的加权和。
"""

from typing import List

from tinytorch.autograd.function import Function
from tinytorch.ndarr import NdArray, Shape


class Conv2d(Function):
    """二维卷积运算。

    对形状为 (N, C, H, W) 的输入执行二维卷积。

    数学表达式:
        output[n, oc, oh, ow] = bias[oc] + Σ_ic Σ_kh Σ_kw input[n, ic, h, w] * weight[oc, ic, kh, kw]

    参数:
        stride: 卷积步长，控制输出特征图的采样密度
        padding: 边缘填充，在输入四周填充零以控制输出尺寸
        kernel_size: 卷积核大小 (正方形)
        use_bias: 是否使用偏置项

    形状说明:
        - 输入:  (N, C_in, H_in, W_in)
        - 权重:  (C_out, C_in, K, K)
        - 偏置:  (C_out,)
        - 输出:  (N, C_out, H_out, W_out)

    其中:
        H_out = (H_in + 2*padding - kernel_size) // stride + 1
        W_out = (W_in + 2*padding - kernel_size) // stride + 1
    """

    def __init__(self, stride: int, padding: int, kernel_size: int, use_bias: bool):
        """初始化二维卷积。

        Args:
            stride: 卷积步长
            padding: 边缘填充像素数
            kernel_size: 卷积核大小 (正方形)
            use_bias: 是否使用偏置项
        """
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.use_bias = use_bias

    @staticmethod
    def _pad_input(input_tensor: NdArray, padding: int) -> NdArray:
        """对输入张量进行零填充。

        Args:
            input_tensor: 输入张量 (N, C, H, W)
            padding: 填充像素数

        Returns:
            填充后的张量 (N, C, H+2*padding, W+2*padding)
        """
        if padding == 0:
            return input_tensor

        batch_size, channels, height, width = input_tensor.shape.dims
        padded_height = height + 2 * padding
        padded_width = width + 2 * padding
        padded_data = [0.0] * (batch_size * channels * padded_height * padded_width)

        # 将原始数据复制到填充后的张量中心位置
        for b in range(batch_size):
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        src_idx = (
                            b * channels * height * width
                            + c * height * width
                            + h * width
                            + w
                        )
                        dst_idx = (
                            b * channels * padded_height * padded_width
                            + c * padded_height * padded_width
                            + (h + padding) * padded_width
                            + (w + padding)
                        )
                        padded_data[dst_idx] = input_tensor.data[src_idx]

        return NdArray(
            padded_data,
            Shape((batch_size, channels, padded_height, padded_width)),
            input_tensor.dtype,
        )

    def forward(self, x: NdArray, weight: NdArray, bias: NdArray = None) -> NdArray:
        """前向传播: 计算二维卷积。

        Args:
            x: 输入张量 (N, C_in, H, W)
            weight: 卷积核权重 (C_out, C_in, K, K)
            bias: 偏置项 (C_out,)，可选

        Returns:
            输出张量 (N, C_out, H_out, W_out)
        """
        self.save_for_backward(x, weight, bias)
        batch_size, in_channels, height, width = x.shape.dims
        out_channels, _, kernel_h, kernel_w = weight.shape.dims

        # 计算输出特征图尺寸
        out_height = (height + 2 * self.padding - kernel_h) // self.stride + 1
        out_width = (width + 2 * self.padding - kernel_w) // self.stride + 1

        # 对输入进行填充
        padded_input = self._pad_input(x, self.padding)
        p_height, p_width = padded_input.shape.dims[2], padded_input.shape.dims[3]
        output_data = []

        # 执行卷积计算
        for b in range(batch_size):
            for oc in range(out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        conv_sum = 0.0

                        # 对每个输入通道和卷积核位置计算加权和
                        for ic in range(in_channels):
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    h_idx = h_start + kh
                                    w_idx = w_start + kw
                                    input_idx = (
                                        b * in_channels * p_height * p_width
                                        + ic * p_height * p_width
                                        + h_idx * p_width
                                        + w_idx
                                    )
                                    weight_idx = (
                                        oc * in_channels * kernel_h * kernel_w
                                        + ic * kernel_h * kernel_w
                                        + kh * kernel_w
                                        + kw
                                    )
                                    conv_sum += padded_input.data[input_idx] * weight.data[weight_idx]

                        # 添加偏置
                        if self.use_bias and bias is not None:
                            conv_sum += bias.data[oc]
                        output_data.append(conv_sum)

        return NdArray(output_data, Shape((batch_size, out_channels, out_height, out_width)), x.dtype)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 计算输入、权重和偏置的梯度。

        Args:
            grad_output: 输出梯度 (N, C_out, H_out, W_out)

        Returns:
            [grad_input, grad_weight, grad_bias] 梯度列表
        """
        x, weight, bias = self.get_saved_tensors()
        batch_size, in_channels, height, width = x.shape.dims
        out_channels, _, kernel_h, kernel_w = weight.shape.dims
        out_height, out_width = grad_output.shape.dims[2], grad_output.shape.dims[3]

        # 获取填充后的输入
        padded_input = self._pad_input(x, self.padding)
        p_height, p_width = padded_input.shape.dims[2], padded_input.shape.dims[3]

        # 初始化梯度
        grad_padded_input = [0.0] * padded_input.shape.size
        grad_weight = [0.0] * weight.shape.size
        grad_bias = [0.0] * out_channels if (self.use_bias and bias is not None) else None

        # 计算梯度
        for b in range(batch_size):
            for oc in range(out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # 获取输出梯度
                        go_idx = (
                            b * out_channels * out_height * out_width
                            + oc * out_height * out_width
                            + oh * out_width
                            + ow
                        )
                        go = grad_output.data[go_idx]

                        # 累加偏置梯度
                        if grad_bias is not None:
                            grad_bias[oc] += go

                        # 计算输入和权重梯度
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        for ic in range(in_channels):
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    h_idx = h_start + kh
                                    w_idx = w_start + kw
                                    input_idx = (
                                        b * in_channels * p_height * p_width
                                        + ic * p_height * p_width
                                        + h_idx * p_width
                                        + w_idx
                                    )
                                    weight_idx = (
                                        oc * in_channels * kernel_h * kernel_w
                                        + ic * kernel_h * kernel_w
                                        + kh * kernel_w
                                        + kw
                                    )
                                    # 输入梯度: 梯度乘以权重
                                    grad_padded_input[input_idx] += go * weight.data[weight_idx]
                                    # 权重梯度: 梯度乘以输入
                                    grad_weight[weight_idx] += go * padded_input.data[input_idx]

        # 移除填充部分，恢复原始输入尺寸的梯度
        if self.padding == 0:
            grad_input_data = grad_padded_input
        else:
            grad_input_data = []
            for b in range(batch_size):
                for c in range(in_channels):
                    for h in range(height):
                        for w in range(width):
                            src_idx = (
                                b * in_channels * p_height * p_width
                                + c * p_height * p_width
                                + (h + self.padding) * p_width
                                + (w + self.padding)
                            )
                            grad_input_data.append(grad_padded_input[src_idx])

        # 构造梯度张量
        grad_input = NdArray(grad_input_data, x.shape, x.dtype)
        grad_weight_arr = NdArray(grad_weight, weight.shape, weight.dtype)

        if grad_bias is not None:
            grad_bias_arr = NdArray(grad_bias, Shape((out_channels,)), x.dtype)
            return [grad_input, grad_weight_arr, grad_bias_arr]

        return [grad_input, grad_weight_arr]
