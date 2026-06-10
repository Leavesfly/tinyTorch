"""Conv2d 性能基准测试。

验证 im2col + 矩阵乘法优化后的 Conv2d 前向/反向传播的正确性和性能。
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinytorch.ndarr import NdArray, Shape
from tinytorch.autograd.ops.conv import Conv2d, _im2col, _col2im


def test_conv2d_forward_correctness():
    """验证 Conv2d 前向传播的数值正确性。"""
    print("=== 测试 Conv2d 前向传播正确性 ===")

    # 简单的 1x1x3x3 输入, 1x1x2x2 卷积核
    input_data = [1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0,
                  7.0, 8.0, 9.0]
    x = NdArray(input_data, Shape((1, 1, 3, 3)))

    weight_data = [1.0, 0.0,
                   0.0, 1.0]
    weight = NdArray(weight_data, Shape((1, 1, 2, 2)))

    bias = NdArray([0.5], Shape((1,)))

    conv = Conv2d(stride=1, padding=0, kernel_size=2, use_bias=True)
    output = conv.forward(x, weight, bias)

    # 手动计算期望值:
    # out[0,0,0,0] = 1*1 + 2*0 + 4*0 + 5*1 + 0.5 = 6.5
    # out[0,0,0,1] = 2*1 + 3*0 + 5*0 + 6*1 + 0.5 = 8.5
    # out[0,0,1,0] = 4*1 + 5*0 + 7*0 + 8*1 + 0.5 = 12.5
    # out[0,0,1,1] = 5*1 + 6*0 + 8*0 + 9*1 + 0.5 = 14.5
    expected = [6.5, 8.5, 12.5, 14.5]

    assert output.shape.dims == (1, 1, 2, 2), f"Shape mismatch: {output.shape.dims}"
    for i, (got, exp) in enumerate(zip(output.data, expected)):
        assert abs(got - exp) < 1e-6, f"Value mismatch at {i}: got {got}, expected {exp}"

    print("  ✅ 基本前向传播正确")


def test_conv2d_with_padding():
    """验证带 padding 的 Conv2d。"""
    print("=== 测试 Conv2d padding ===")

    x = NdArray([1.0, 2.0, 3.0, 4.0], Shape((1, 1, 2, 2)))
    weight = NdArray([1.0, 1.0, 1.0, 1.0], Shape((1, 1, 2, 2)))
    bias = NdArray([0.0], Shape((1,)))

    conv = Conv2d(stride=1, padding=1, kernel_size=2, use_bias=True)
    output = conv.forward(x, weight, bias)

    # padding=1 后输入变为 4x4 (全零边框)，输出应为 3x3
    assert output.shape.dims == (1, 1, 3, 3), f"Shape mismatch: {output.shape.dims}"

    # out[0,0,0,0] = 0+0+0+1 = 1.0
    assert abs(output.data[0] - 1.0) < 1e-6, f"Corner value wrong: {output.data[0]}"
    # out[0,0,1,1] = 1+2+3+4 = 10.0
    assert abs(output.data[4] - 10.0) < 1e-6, f"Center value wrong: {output.data[4]}"

    print("  ✅ Padding 正确")


def test_conv2d_backward_correctness():
    """验证 Conv2d 反向传播的梯度正确性。"""
    print("=== 测试 Conv2d 反向传播正确性 ===")

    x = NdArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], Shape((1, 1, 3, 3)))
    weight = NdArray([1.0, 0.0, 0.0, 1.0], Shape((1, 1, 2, 2)))
    bias = NdArray([0.5], Shape((1,)))

    conv = Conv2d(stride=1, padding=0, kernel_size=2, use_bias=True)
    output = conv.forward(x, weight, bias)

    # 全 1 梯度
    grad_output = NdArray([1.0, 1.0, 1.0, 1.0], Shape((1, 1, 2, 2)))
    grads = conv.backward(grad_output)

    grad_input, grad_weight, grad_bias = grads

    # 验证梯度形状
    assert grad_input.shape.dims == (1, 1, 3, 3), f"grad_input shape: {grad_input.shape.dims}"
    assert grad_weight.shape.dims == (1, 1, 2, 2), f"grad_weight shape: {grad_weight.shape.dims}"
    assert grad_bias.shape.dims == (1,), f"grad_bias shape: {grad_bias.shape.dims}"

    # 偏置梯度 = sum(grad_output) = 4.0
    assert abs(grad_bias.data[0] - 4.0) < 1e-6, f"grad_bias wrong: {grad_bias.data[0]}"

    print("  ✅ 反向传播梯度正确")


def test_conv2d_multi_channel():
    """验证多通道 Conv2d。"""
    print("=== 测试多通道 Conv2d ===")

    batch_size, in_channels, height, width = 2, 3, 4, 4
    out_channels, kernel_size = 2, 3

    total_input = batch_size * in_channels * height * width
    x = NdArray([float(i % 7) for i in range(total_input)], Shape((batch_size, in_channels, height, width)))

    total_weight = out_channels * in_channels * kernel_size * kernel_size
    weight = NdArray([float(i % 5) * 0.1 for i in range(total_weight)], Shape((out_channels, in_channels, kernel_size, kernel_size)))

    bias = NdArray([0.1, -0.1], Shape((out_channels,)))

    conv = Conv2d(stride=1, padding=1, kernel_size=kernel_size, use_bias=True)
    output = conv.forward(x, weight, bias)

    expected_out_h = (height + 2 * 1 - kernel_size) // 1 + 1
    expected_out_w = (width + 2 * 1 - kernel_size) // 1 + 1
    assert output.shape.dims == (batch_size, out_channels, expected_out_h, expected_out_w), \
        f"Shape mismatch: {output.shape.dims}"

    # 验证反向传播
    grad_output = NdArray([1.0] * output.shape.size, output.shape)
    grads = conv.backward(grad_output)
    assert grads[0].shape.dims == x.shape.dims
    assert grads[1].shape.dims == weight.shape.dims
    assert grads[2].shape.dims == (out_channels,)

    print(f"  ✅ 多通道 Conv2d 正确 (output shape: {output.shape.dims})")


def benchmark_conv2d():
    """Conv2d 性能基准测试。"""
    print("\n=== Conv2d 性能基准测试 ===")

    configs = [
        {"name": "小规模 (1,1,8,8) k=3", "batch": 1, "ic": 1, "h": 8, "w": 8, "oc": 1, "k": 3, "p": 1},
        {"name": "中规模 (2,3,16,16) k=3", "batch": 2, "ic": 3, "h": 16, "w": 16, "oc": 8, "k": 3, "p": 1},
        {"name": "较大规模 (1,3,32,32) k=3", "batch": 1, "ic": 3, "h": 32, "w": 32, "oc": 16, "k": 3, "p": 1},
    ]

    for cfg in configs:
        total_input = cfg["batch"] * cfg["ic"] * cfg["h"] * cfg["w"]
        x = NdArray([float(i % 10) * 0.1 for i in range(total_input)],
                    Shape((cfg["batch"], cfg["ic"], cfg["h"], cfg["w"])))

        total_weight = cfg["oc"] * cfg["ic"] * cfg["k"] * cfg["k"]
        weight = NdArray([float(i % 7) * 0.1 for i in range(total_weight)],
                         Shape((cfg["oc"], cfg["ic"], cfg["k"], cfg["k"])))

        bias = NdArray([0.0] * cfg["oc"], Shape((cfg["oc"],)))

        conv = Conv2d(stride=1, padding=cfg["p"], kernel_size=cfg["k"], use_bias=True)

        # 前向传播计时
        start = time.time()
        output = conv.forward(x, weight, bias)
        forward_time = time.time() - start

        # 反向传播计时
        grad_output = NdArray([1.0] * output.shape.size, output.shape)
        start = time.time()
        conv.backward(grad_output)
        backward_time = time.time() - start

        print(f"  {cfg['name']}:")
        print(f"    前向: {forward_time*1000:.2f}ms | 反向: {backward_time*1000:.2f}ms")


if __name__ == "__main__":
    test_conv2d_forward_correctness()
    test_conv2d_with_padding()
    test_conv2d_backward_correctness()
    test_conv2d_multi_channel()
    benchmark_conv2d()
    print("\n🎉 所有 Conv2d 测试通过!")
