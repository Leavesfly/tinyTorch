"""测试高级神经网络层：Conv2d、Embedding、MultiHeadAttention。

以及本次修复的关键功能的回归测试。

Author: TinyAI Team
"""

import math
import pytest
from tinytorch.nn.layers.conv import Conv2d
from tinytorch.nn.layers.normalization import Embedding, Dropout, LayerNorm
from tinytorch.nn.layers.attention import MultiHeadAttention
from tinytorch.autograd import Tensor
from tinytorch.ndarr import NdArray, Shape
from tinytorch.utils import random as tt_random


class TestConv2d:
    """Conv2d 层的测试。"""

    def test_conv2d_creation(self):
        """测试 Conv2d 层创建和属性。"""
        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        assert conv.in_channels == 3
        assert conv.out_channels == 16
        assert conv.kernel_size == 3
        assert conv.stride == 1
        assert conv.padding == 1
        assert conv.weight is not None
        assert conv.bias is not None

    def test_conv2d_no_bias(self):
        """测试无偏置的 Conv2d。"""
        conv = Conv2d(in_channels=1, out_channels=4, kernel_size=2, use_bias=False)
        assert conv.bias is None

    def test_conv2d_weight_shape(self):
        """测试权重形状正确性。"""
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=5)
        assert conv.weight.value.shape.dims == (8, 3, 5, 5)

    def test_conv2d_forward_shape(self):
        """测试前向传播输出形状。"""
        tt_random.seed(42)
        conv = Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0)
        x = Tensor(NdArray.randn((1, 1, 5, 5)), requires_grad=False)
        output = conv(x)
        # output_size = (5 - 3 + 2*0) / 1 + 1 = 3
        assert output.value.shape.dims == (1, 2, 3, 3)

    def test_conv2d_forward_with_padding(self):
        """测试带 padding 的前向传播。"""
        tt_random.seed(42)
        conv = Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        x = Tensor(NdArray.randn((1, 1, 4, 4)), requires_grad=False)
        output = conv(x)
        # output_size = (4 - 3 + 2*1) / 1 + 1 = 4
        assert output.value.shape.dims == (1, 2, 4, 4)

    def test_conv2d_input_channel_mismatch(self):
        """测试输入通道不匹配时抛出异常。"""
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        x = Tensor(NdArray.randn((1, 1, 5, 5)), requires_grad=False)
        with pytest.raises(ValueError, match="Expected 3 input channels"):
            conv(x)

    def test_conv2d_repr(self):
        """测试字符串表示。"""
        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        repr_str = repr(conv)
        assert "Conv2d" in repr_str
        assert "in_channels=3" in repr_str
        assert "out_channels=16" in repr_str


class TestEmbedding:
    """Embedding 层的测试。"""

    def test_embedding_creation(self):
        """测试 Embedding 层创建。"""
        emb = Embedding(num_embeddings=100, embedding_dim=32)
        assert emb.num_embeddings == 100
        assert emb.embedding_dim == 32
        assert emb.weight.value.shape.dims == (100, 32)

    def test_embedding_forward_2d(self):
        """测试 2D 输入的前向传播。"""
        tt_random.seed(42)
        emb = Embedding(num_embeddings=10, embedding_dim=4)
        indices = Tensor(NdArray([[1, 2, 3], [4, 5, 6]]), requires_grad=False)
        output = emb(indices)
        assert output.value.shape.dims == (2, 3, 4)

    def test_embedding_padding_idx(self):
        """测试 padding_idx 功能。"""
        tt_random.seed(42)
        emb = Embedding(num_embeddings=10, embedding_dim=4, padding_idx=0)
        # 验证 padding_idx 对应的嵌入向量为零
        for j in range(4):
            assert emb.weight.value.data[j] == 0.0

    def test_embedding_padding_idx_invalid(self):
        """测试无效的 padding_idx 抛出异常。"""
        with pytest.raises(ValueError, match="padding_idx must be in"):
            Embedding(num_embeddings=10, embedding_dim=4, padding_idx=10)

    def test_embedding_repr(self):
        """测试字符串表示。"""
        emb = Embedding(num_embeddings=1000, embedding_dim=128)
        repr_str = repr(emb)
        assert "Embedding" in repr_str
        assert "1000" in repr_str
        assert "128" in repr_str


class TestMultiHeadAttention:
    """MultiHeadAttention 层的测试。"""

    def test_attention_creation(self):
        """测试 MultiHeadAttention 创建。"""
        attn = MultiHeadAttention(embed_dim=8, num_heads=2)
        assert attn.embed_dim == 8
        assert attn.num_heads == 2
        assert attn.head_dim == 4

    def test_attention_embed_dim_not_divisible(self):
        """测试 embed_dim 不能被 num_heads 整除时抛出异常。"""
        with pytest.raises(ValueError, match="must be divisible"):
            MultiHeadAttention(embed_dim=7, num_heads=2)

    def test_attention_self_attention_shape(self):
        """测试自注意力输出形状。"""
        tt_random.seed(42)
        attn = MultiHeadAttention(embed_dim=8, num_heads=2, dropout=0.0)
        x = Tensor(NdArray.randn((1, 3, 8)), requires_grad=False)
        output = attn(x)
        assert output.value.shape.dims == (1, 3, 8)

    def test_attention_repr(self):
        """测试字符串表示。"""
        attn = MultiHeadAttention(embed_dim=512, num_heads=8)
        repr_str = repr(attn)
        assert "MultiHeadAttention" in repr_str
        assert "512" in repr_str
        assert "8" in repr_str


class TestBroadcastFix:
    """测试 Shape.can_broadcast 修复。"""

    def test_can_broadcast_different_ndim(self):
        """测试不同维度数的广播判断。"""
        s1 = Shape((3, 4))
        s2 = Shape((2, 3, 4))
        # (3, 4) 广播到 (2, 3, 4)：左侧补 1 → (1, 3, 4) vs (2, 3, 4) → 兼容
        assert s1.can_broadcast(s2) is True

    def test_can_broadcast_incompatible(self):
        """测试不兼容的广播。"""
        s1 = Shape((3, 5))
        s2 = Shape((2, 3, 4))
        # (3, 5) → (1, 3, 5) vs (2, 3, 4) → 5 != 4 且都不为 1 → 不兼容
        assert s1.can_broadcast(s2) is False

    def test_can_broadcast_scalar(self):
        """测试标量广播。"""
        s1 = Shape((1,))
        s2 = Shape((3, 4, 5))
        assert s1.can_broadcast(s2) is True

    def test_can_broadcast_same_shape(self):
        """测试相同形状。"""
        s1 = Shape((3, 4))
        s2 = Shape((3, 4))
        assert s1.can_broadcast(s2) is True


class TestLogSqrtNaN:
    """测试 log/sqrt 对非法输入返回 NaN 而非抛异常。"""

    def test_log_negative(self):
        """测试 log 对负数返回 NaN。"""
        arr = NdArray([-1.0, -2.0])
        result = arr.log()
        assert math.isnan(result.data[0])
        assert math.isnan(result.data[1])

    def test_log_zero(self):
        """测试 log(0) 返回 -inf。"""
        arr = NdArray([0.0])
        result = arr.log()
        assert result.data[0] == float('-inf')

    def test_log_positive(self):
        """测试 log 对正数正常工作。"""
        arr = NdArray([1.0, math.e])
        result = arr.log()
        assert abs(result.data[0] - 0.0) < 1e-10
        assert abs(result.data[1] - 1.0) < 1e-10

    def test_sqrt_negative(self):
        """测试 sqrt 对负数返回 NaN。"""
        arr = NdArray([-4.0])
        result = arr.sqrt()
        assert math.isnan(result.data[0])

    def test_sqrt_positive(self):
        """测试 sqrt 对正数正常工作。"""
        arr = NdArray([4.0, 9.0])
        result = arr.sqrt()
        assert abs(result.data[0] - 2.0) < 1e-10
        assert abs(result.data[1] - 3.0) < 1e-10


class TestSigmoidStability:
    """测试 sigmoid 数值稳定性。"""

    def test_sigmoid_extreme_positive(self):
        """测试极大正数不溢出。"""
        arr = NdArray([1000.0, 710.0])
        result = arr.sigmoid()
        assert result.data[0] == 1.0
        assert result.data[1] == 1.0

    def test_sigmoid_extreme_negative(self):
        """测试极大负数不溢出。"""
        arr = NdArray([-1000.0, -750.0])
        result = arr.sigmoid()
        assert result.data[0] == 0.0
        assert result.data[1] == 0.0

    def test_sigmoid_normal(self):
        """测试正常范围的 sigmoid。"""
        arr = NdArray([0.0])
        result = arr.sigmoid()
        assert abs(result.data[0] - 0.5) < 1e-10


class TestReverseOperators:
    """测试 NdArray 反向运算符。"""

    def test_radd(self):
        """测试 scalar + NdArray。"""
        arr = NdArray([1.0, 2.0, 3.0])
        result = 10 + arr
        assert result.data == [11.0, 12.0, 13.0]

    def test_rsub(self):
        """测试 scalar - NdArray。"""
        arr = NdArray([1.0, 2.0, 3.0])
        result = 10 - arr
        assert result.data == [9.0, 8.0, 7.0]

    def test_rmul(self):
        """测试 scalar * NdArray。"""
        arr = NdArray([1.0, 2.0, 3.0])
        result = 5 * arr
        assert result.data == [5.0, 10.0, 15.0]

    def test_rtruediv(self):
        """测试 scalar / NdArray。"""
        arr = NdArray([2.0, 4.0, 5.0])
        result = 10 / arr
        assert result.data == [5.0, 2.5, 2.0]


class TestDtypeConsistency:
    """测试归约运算的 dtype 一致性。"""

    def test_sum_int32(self):
        """测试 int32 类型的 sum 初始值。"""
        arr = NdArray([1, 2, 3], dtype='int32')
        result = arr.sum()
        assert result.data[0] == 6
