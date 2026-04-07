"""卷积运算 - 支持自动微分。

本模块实现二维卷积运算 Conv2d，用于构建卷积神经网络。

使用 im2col/col2im + 矩阵乘法实现高效卷积，
将卷积操作转换为矩阵乘法，避免多重 Python 循环带来的性能瓶颈。
"""

from typing import List

from tinytorch.autograd.function import Function
from tinytorch.ndarr import NdArray, Shape


def _im2col(input_data: list, batch_size: int, in_channels: int,
            input_height: int, input_width: int,
            kernel_h: int, kernel_w: int,
            stride: int, padding: int) -> tuple:
    """将输入数据展开为列矩阵 (im2col)。

    将每个滑动窗口展开为一列，使卷积运算可以转化为矩阵乘法。

    Args:
        input_data: 输入的扁平数据列表
        batch_size: 批次大小 N
        in_channels: 输入通道数 C_in
        input_height: 输入高度 H
        input_width: 输入宽度 W
        kernel_h: 卷积核高度
        kernel_w: 卷积核宽度
        stride: 步长
        padding: 填充大小

    Returns:
        (col_data, col_row_size, col_col_size) 元组:
        - col_data: 展开后的扁平列表，逻辑形状 (N, C_in*K_h*K_w, out_h*out_w)
        - col_row_size: C_in*K_h*K_w
        - col_col_size: out_h*out_w
    """
    padded_h = input_height + 2 * padding
    padded_w = input_width + 2 * padding
    out_h = (padded_h - kernel_h) // stride + 1
    out_w = (padded_w - kernel_w) // stride + 1

    col_row_size = in_channels * kernel_h * kernel_w
    col_col_size = out_h * out_w
    col_data = [0.0] * (batch_size * col_row_size * col_col_size)

    channel_stride_in = input_height * input_width
    batch_stride_in = in_channels * channel_stride_in
    batch_stride_col = col_row_size * col_col_size

    for b in range(batch_size):
        b_offset_in = b * batch_stride_in
        b_offset_col = b * batch_stride_col
        col_col_idx = 0
        for oh in range(out_h):
            h_start = oh * stride - padding
            for ow in range(out_w):
                w_start = ow * stride - padding
                col_row_idx = 0
                for ic in range(in_channels):
                    ic_offset = b_offset_in + ic * channel_stride_in
                    for kh in range(kernel_h):
                        h_idx = h_start + kh
                        for kw in range(kernel_w):
                            w_idx = w_start + kw
                            if 0 <= h_idx < input_height and 0 <= w_idx < input_width:
                                col_data[b_offset_col + col_row_idx * col_col_size + col_col_idx] = \
                                    input_data[ic_offset + h_idx * input_width + w_idx]
                            col_row_idx += 1
                col_col_idx += 1

    return col_data, col_row_size, col_col_size


def _col2im(col_data: list, batch_size: int, in_channels: int,
            input_height: int, input_width: int,
            kernel_h: int, kernel_w: int,
            stride: int, padding: int,
            col_row_size: int, col_col_size: int) -> list:
    """将列矩阵还原为输入梯度 (col2im)。

    im2col 的逆操作，将列矩阵中的梯度累加回原始输入位置。

    Args:
        col_data: 列矩阵的扁平数据
        batch_size: 批次大小
        in_channels: 输入通道数
        input_height: 原始输入高度
        input_width: 原始输入宽度
        kernel_h: 卷积核高度
        kernel_w: 卷积核宽度
        stride: 步长
        padding: 填充大小
        col_row_size: 列矩阵的行数 (C_in*K_h*K_w)
        col_col_size: 列矩阵的列数 (out_h*out_w)

    Returns:
        还原后的输入梯度扁平列表，形状为 (N, C_in, H, W)
    """
    out_h = (input_height + 2 * padding - kernel_h) // stride + 1
    out_w = (input_width + 2 * padding - kernel_w) // stride + 1

    channel_stride = input_height * input_width
    batch_stride_in = in_channels * channel_stride
    batch_stride_col = col_row_size * col_col_size

    grad_input_data = [0.0] * (batch_size * batch_stride_in)

    for b in range(batch_size):
        b_offset_in = b * batch_stride_in
        b_offset_col = b * batch_stride_col
        col_col_idx = 0
        for oh in range(out_h):
            h_start = oh * stride - padding
            for ow in range(out_w):
                w_start = ow * stride - padding
                col_row_idx = 0
                for ic in range(in_channels):
                    ic_offset = b_offset_in + ic * channel_stride
                    for kh in range(kernel_h):
                        h_idx = h_start + kh
                        for kw in range(kernel_w):
                            w_idx = w_start + kw
                            if 0 <= h_idx < input_height and 0 <= w_idx < input_width:
                                grad_input_data[ic_offset + h_idx * input_width + w_idx] += \
                                    col_data[b_offset_col + col_row_idx * col_col_size + col_col_idx]
                            col_row_idx += 1
                col_col_idx += 1

    return grad_input_data


class Conv2d(Function):
    """二维卷积运算 (im2col + 矩阵乘法实现)。

    对形状为 (N, C, H, W) 的输入执行二维卷积。

    优化策略:
        使用 im2col 将卷积转换为矩阵乘法:
        1. im2col: 将输入的每个滑动窗口展开为一列 -> (C_in*K*K, out_h*out_w)
        2. 权重 reshape: (C_out, C_in*K*K)
        3. 矩阵乘法: weight_matrix @ col_matrix -> (C_out, out_h*out_w)
        4. reshape 回 (N, C_out, out_h, out_w)

    数学表达式:
        output[n, oc, oh, ow] = bias[oc] + sum_{ic,kh,kw} input[n, ic, h, w] * weight[oc, ic, kh, kw]

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

    def forward(self, x: NdArray, weight: NdArray, bias: NdArray = None) -> NdArray:
        """前向传播: 使用 im2col + 矩阵乘法计算二维卷积。

        将卷积操作转换为矩阵乘法，避免 7 重 Python 循环:
        1. im2col 展开输入 -> col_matrix (C_in*K*K, out_h*out_w) per batch
        2. weight reshape -> weight_matrix (C_out, C_in*K*K)
        3. output = weight_matrix @ col_matrix -> (C_out, out_h*out_w) per batch

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

        out_height = (height + 2 * self.padding - kernel_h) // self.stride + 1
        out_width = (width + 2 * self.padding - kernel_w) // self.stride + 1

        # im2col: 将输入展开为列矩阵
        col_data, col_row_size, col_col_size = _im2col(
            x.data, batch_size, in_channels, height, width,
            kernel_h, kernel_w, self.stride, self.padding
        )

        # 将权重 reshape 为 2D 矩阵: (C_out, C_in*K_h*K_w)
        weight_matrix = NdArray(weight.data, Shape((out_channels, col_row_size)), weight.dtype)

        # 对每个 batch 执行矩阵乘法
        output_size = batch_size * out_channels * out_height * out_width
        output_data = [0.0] * output_size
        batch_col_stride = col_row_size * col_col_size
        batch_out_stride = out_channels * col_col_size

        for b in range(batch_size):
            # 提取当前 batch 的 col 矩阵
            col_start = b * batch_col_stride
            col_end = col_start + batch_col_stride
            col_matrix = NdArray(
                col_data[col_start:col_end],
                Shape((col_row_size, col_col_size)),
                x.dtype
            )

            # 矩阵乘法: (C_out, C_in*K*K) @ (C_in*K*K, out_h*out_w) -> (C_out, out_h*out_w)
            result = weight_matrix.matmul(col_matrix)

            # 复制结果到输出
            out_start = b * batch_out_stride
            output_data[out_start:out_start + batch_out_stride] = result.data

        # 添加偏置: 对每个输出通道加上对应的偏置值
        if self.use_bias and bias is not None:
            for b in range(batch_size):
                for oc in range(out_channels):
                    bias_val = bias.data[oc]
                    offset = b * batch_out_stride + oc * col_col_size
                    for i in range(col_col_size):
                        output_data[offset + i] += bias_val

        # 保存 im2col 结果供反向传播使用，避免重复计算
        self._col_data = col_data
        self._col_row_size = col_row_size
        self._col_col_size = col_col_size

        return NdArray(
            output_data,
            Shape((batch_size, out_channels, out_height, out_width)),
            x.dtype
        )

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        """反向传播: 使用矩阵乘法 + col2im 计算梯度。

        梯度计算策略:
        1. 权重梯度: grad_weight = grad_output_2d @ col_matrix^T (矩阵乘法)
        2. 输入梯度: grad_col = weight^T @ grad_output_2d, 然后 col2im 还原
        3. 偏置梯度: grad_bias = sum(grad_output, axis=[0,2,3])

        Args:
            grad_output: 输出梯度 (N, C_out, H_out, W_out)

        Returns:
            [grad_input, grad_weight, grad_bias] 梯度列表
        """
        x, weight, bias = self.get_saved_tensors()
        batch_size, in_channels, height, width = x.shape.dims
        out_channels, _, kernel_h, kernel_w = weight.shape.dims
        out_height, out_width = grad_output.shape.dims[2], grad_output.shape.dims[3]

        col_data = self._col_data
        col_row_size = self._col_row_size
        col_col_size = self._col_col_size

        # 将权重 reshape 为 2D: (C_out, C_in*K*K)
        weight_matrix = NdArray(weight.data, Shape((out_channels, col_row_size)), weight.dtype)
        # 权重转置: (C_in*K*K, C_out)
        weight_matrix_t = weight_matrix.transpose()

        batch_out_stride = out_channels * col_col_size
        batch_col_stride = col_row_size * col_col_size

        # 初始化权重梯度累加器
        grad_weight_data = [0.0] * (out_channels * col_row_size)

        # 初始化输入梯度的 col 数据
        grad_col_all = [0.0] * (batch_size * batch_col_stride)

        # 偏置梯度
        grad_bias_data = None
        if self.use_bias and bias is not None:
            grad_bias_data = [0.0] * out_channels

        for b in range(batch_size):
            # 提取当前 batch 的 grad_output，reshape 为 (C_out, out_h*out_w)
            go_start = b * batch_out_stride
            go_end = go_start + batch_out_stride
            grad_out_matrix = NdArray(
                grad_output.data[go_start:go_end],
                Shape((out_channels, col_col_size)),
                grad_output.dtype
            )

            # 提取当前 batch 的 col 矩阵: (C_in*K*K, out_h*out_w)
            col_start = b * batch_col_stride
            col_end = col_start + batch_col_stride
            col_matrix = NdArray(
                col_data[col_start:col_end],
                Shape((col_row_size, col_col_size)),
                x.dtype
            )
            # col 转置: (out_h*out_w, C_in*K*K)
            col_matrix_t = col_matrix.transpose()

            # 权重梯度: (C_out, out_h*out_w) @ (out_h*out_w, C_in*K*K) -> (C_out, C_in*K*K)
            grad_w_batch = grad_out_matrix.matmul(col_matrix_t)
            for i in range(len(grad_weight_data)):
                grad_weight_data[i] += grad_w_batch.data[i]

            # 输入梯度: (C_in*K*K, C_out) @ (C_out, out_h*out_w) -> (C_in*K*K, out_h*out_w)
            grad_col_batch = weight_matrix_t.matmul(grad_out_matrix)
            gc_start = b * batch_col_stride
            grad_col_all[gc_start:gc_start + batch_col_stride] = grad_col_batch.data

            # 偏置梯度: 对 grad_output 沿空间维度求和
            if grad_bias_data is not None:
                for oc in range(out_channels):
                    oc_offset = oc * col_col_size
                    for i in range(col_col_size):
                        grad_bias_data[oc] += grad_out_matrix.data[oc_offset + i]

        # col2im: 将列矩阵梯度还原为输入梯度
        grad_input_data = _col2im(
            grad_col_all, batch_size, in_channels, height, width,
            kernel_h, kernel_w, self.stride, self.padding,
            col_row_size, col_col_size
        )

        # 构造梯度张量
        grad_input = NdArray(grad_input_data, x.shape, x.dtype)
        grad_weight_arr = NdArray(
            grad_weight_data,
            weight.shape,
            weight.dtype
        )

        # 清理缓存的 im2col 数据，释放内存
        del self._col_data
        del self._col_row_size
        del self._col_col_size

        if grad_bias_data is not None:
            grad_bias_arr = NdArray(grad_bias_data, Shape((out_channels,)), x.dtype)
            return [grad_input, grad_weight_arr, grad_bias_arr]

        return [grad_input, grad_weight_arr]
