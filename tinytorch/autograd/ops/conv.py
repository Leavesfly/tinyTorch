"""Convolution autograd operations."""

from typing import List

from tinytorch.autograd.function import Function
from tinytorch.ndarr import NdArray, Shape


class Conv2d(Function):
    """2D convolution over ``(N, C, H, W)`` inputs."""

    def __init__(self, stride: int, padding: int, kernel_size: int, use_bias: bool):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.use_bias = use_bias

    @staticmethod
    def _pad_input(input_tensor: NdArray, padding: int) -> NdArray:
        if padding == 0:
            return input_tensor

        batch_size, channels, height, width = input_tensor.shape.dims
        padded_height = height + 2 * padding
        padded_width = width + 2 * padding
        padded_data = [0.0] * (batch_size * channels * padded_height * padded_width)

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
        self.save_for_backward(x, weight, bias)
        batch_size, in_channels, height, width = x.shape.dims
        out_channels, _, kernel_h, kernel_w = weight.shape.dims

        out_height = (height + 2 * self.padding - kernel_h) // self.stride + 1
        out_width = (width + 2 * self.padding - kernel_w) // self.stride + 1

        padded_input = self._pad_input(x, self.padding)
        p_height, p_width = padded_input.shape.dims[2], padded_input.shape.dims[3]
        output_data = []

        for b in range(batch_size):
            for oc in range(out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        conv_sum = 0.0
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
                        if self.use_bias and bias is not None:
                            conv_sum += bias.data[oc]
                        output_data.append(conv_sum)

        return NdArray(output_data, Shape((batch_size, out_channels, out_height, out_width)), x.dtype)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        x, weight, bias = self.get_saved_tensors()
        batch_size, in_channels, height, width = x.shape.dims
        out_channels, _, kernel_h, kernel_w = weight.shape.dims
        out_height, out_width = grad_output.shape.dims[2], grad_output.shape.dims[3]

        padded_input = self._pad_input(x, self.padding)
        p_height, p_width = padded_input.shape.dims[2], padded_input.shape.dims[3]
        grad_padded_input = [0.0] * padded_input.shape.size
        grad_weight = [0.0] * weight.shape.size
        grad_bias = [0.0] * out_channels if (self.use_bias and bias is not None) else None

        for b in range(batch_size):
            for oc in range(out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        go_idx = (
                            b * out_channels * out_height * out_width
                            + oc * out_height * out_width
                            + oh * out_width
                            + ow
                        )
                        go = grad_output.data[go_idx]

                        if grad_bias is not None:
                            grad_bias[oc] += go

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
                                    grad_padded_input[input_idx] += go * weight.data[weight_idx]
                                    grad_weight[weight_idx] += go * padded_input.data[input_idx]

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

        grad_input = NdArray(grad_input_data, x.shape, x.dtype)
        grad_weight_arr = NdArray(grad_weight, weight.shape, weight.dtype)

        if grad_bias is not None:
            grad_bias_arr = NdArray(grad_bias, Shape((out_channels,)), x.dtype)
            return [grad_input, grad_weight_arr, grad_bias_arr]

        return [grad_input, grad_weight_arr]
