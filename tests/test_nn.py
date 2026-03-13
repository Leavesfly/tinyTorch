"""测试神经网络层。

Author: TinyAI Team
"""

import math

import pytest
from tinytorch.nn.layers import Linear, ReLU, LeakyReLU, LayerNorm, Dropout
from tinytorch.nn.parameter import Parameter
from tinytorch.nn.module import Module
from tinytorch.autograd import Tensor
from tinytorch.ndarr import NdArray
from tinytorch.utils import random as tt_random


class TestLinear:
    """Linear 层的测试。"""
    
    def test_linear_creation(self):
        """测试全连接层创建。"""
        layer = Linear(10, 5)
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.weight is not None
        
    def test_linear_forward(self):
        """测试全连接层前向传播。"""
        layer = Linear(3, 2)
        x = Tensor(NdArray([[1.0, 2.0, 3.0]]))
        output = layer(x)
        assert output.value.shape.dims[0] == 1
        assert output.value.shape.dims[1] == 2

    def test_linear_3d_backward(self):
        """测试 3D 输入时梯度可回传。"""
        layer = Linear(3, 2)
        x = Tensor(NdArray([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [1.0, 3.0, 5.0]],
        ]))
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.weight.grad is not None
        if layer.bias is not None:
            assert layer.bias.grad is not None


class TestActivation:
    """激活函数的测试。"""
    
    def test_relu(self):
        """测试 ReLU 激活。"""
        relu = ReLU()
        x = Tensor(NdArray([-1.0, 0.0, 1.0, 2.0]))
        output = relu(x)
        # ReLU 应该将负数变为0
        assert output.value.data[0] == 0.0
        assert output.value.data[3] == 2.0

    def test_leaky_relu(self):
        """测试 LeakyReLU 激活。"""
        act = LeakyReLU(negative_slope=0.1)
        x = Tensor(NdArray([-2.0, 0.0, 3.0]))
        output = act(x)
        assert abs(output.value.data[0] - (-0.2)) < 1e-9
        assert abs(output.value.data[1] - 0.0) < 1e-9
        assert abs(output.value.data[2] - 3.0) < 1e-9


class TestNormalization:
    """归一化层的测试。"""
    
    def test_layer_norm_creation(self):
        """测试 LayerNorm 创建。"""
        ln = LayerNorm((10,))
        assert ln.normalized_shape == (10,)
        assert ln.eps == 1e-5
        
    def test_dropout_creation(self):
        """测试 Dropout 创建。"""
        dropout = Dropout(p=0.5)
        assert dropout.p == 0.5

    def test_dropout_is_reproducible_with_shared_seed(self):
        """测试 Dropout 使用共享 RNG 时可通过 seed 复现。"""
        dropout = Dropout(p=0.5)
        x = Tensor(NdArray([1.0, 2.0, 3.0, 4.0]))

        tt_random.seed(123)
        y1 = dropout(x)
        tt_random.seed(123)
        y2 = dropout(x)

        assert y1.value.data == y2.value.data

    def test_layer_norm_last_dim(self):
        """测试 LayerNorm 按最后维度归一化。"""
        ln = LayerNorm((2,), elementwise_affine=False)
        x = Tensor(NdArray([[1.0, 3.0], [5.0, 7.0]]))
        y = ln(x)

        # 每一行均值应接近 0
        row0_mean = (y.value.data[0] + y.value.data[1]) / 2.0
        row1_mean = (y.value.data[2] + y.value.data[3]) / 2.0
        assert abs(row0_mean) < 1e-5
        assert abs(row1_mean) < 1e-5


class TestConvolution:
    """卷积层的测试。"""
    
    def test_conv2d_creation(self):
        """测试卷积层创建。"""
        try:
            from tinytorch.nn.layers import Conv2d
            conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
            assert conv.in_channels == 3
            assert conv.out_channels == 16
        except ImportError:
            pytest.skip("Conv2d 暂未实现")

    def test_conv2d_backward(self):
        """测试 Conv2d 权重可以反向传播。"""
        try:
            from tinytorch.nn.layers import Conv2d
            conv = Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
            x = Tensor(NdArray.randn((1, 1, 4, 4)))
            y = conv(x)
            loss = y.sum()
            loss.backward()
            assert x.grad is not None
            assert conv.weight.grad is not None
            if conv.bias is not None:
                assert conv.bias.grad is not None
        except ImportError:
            pytest.skip("Conv2d 暂未实现")

    def test_conv2d_forward_matches_fixed_kernel(self):
        """测试 Conv2d 在固定输入和卷积核下给出精确结果。"""
        from tinytorch.nn.layers import Conv2d

        conv = Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
        conv.weight.value = NdArray([[[[1.0, 0.0], [0.0, -1.0]]]])
        conv.bias.value = NdArray([0.0])

        x = Tensor(NdArray([[[[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]]]]))
        y = conv(x)

        assert y.value.shape.dims == (1, 1, 2, 2)
        assert y.value.data == [-4.0, -4.0, -4.0, -4.0]

    def test_conv2d_backward_matches_expected_gradients(self):
        """测试 Conv2d 在固定样例下的输入和参数梯度。"""
        from tinytorch.nn.layers import Conv2d

        conv = Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
        conv.weight.value = NdArray([[[[1.0, 0.0], [0.0, -1.0]]]])
        conv.bias.value = NdArray([0.0])

        x = Tensor(NdArray([[[[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]]]]))
        y = conv(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert conv.weight.grad is not None
        assert conv.bias.grad is not None

        assert x.grad.data == [1.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, -1.0]
        assert conv.weight.grad.data == [12.0, 16.0, 24.0, 28.0]
        assert conv.bias.grad.data == [4.0]


class TestEmbedding:
    """嵌入层的测试。"""
    
    def test_embedding_creation(self):
        """测试嵌入层创建。"""
        try:
            from tinytorch.nn.layers import Embedding
            emb = Embedding(num_embeddings=1000, embedding_dim=128)
            assert emb.num_embeddings == 1000
            assert emb.embedding_dim == 128
        except ImportError:
            pytest.skip("Embedding 暂未实现")

    def test_embedding_weight_backward(self):
        """测试 Embedding 权重可以接收梯度。"""
        try:
            from tinytorch.nn.layers import Embedding
            emb = Embedding(num_embeddings=8, embedding_dim=4)
            indices = Tensor(NdArray([[1, 2], [2, 3]]), requires_grad=False)
            out = emb(indices)
            loss = out.sum()
            loss.backward()
            assert emb.weight.grad is not None
            # 索引 2 出现两次，其梯度和应大于单次索引
            row2_grad_sum = sum(emb.weight.grad.data[2 * 4: 3 * 4])
            row1_grad_sum = sum(emb.weight.grad.data[1 * 4: 2 * 4])
            assert row2_grad_sum > row1_grad_sum
        except ImportError:
            pytest.skip("Embedding 暂未实现")

    def test_embedding_forward_matches_expected_values(self):
        """测试 Embedding 在固定权重下返回正确向量。"""
        from tinytorch.nn.layers import Embedding

        emb = Embedding(num_embeddings=4, embedding_dim=2)
        emb.weight.value = NdArray([[9.0, 9.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        indices = Tensor(NdArray([[0, 1, 3]]), requires_grad=False)
        out = emb(indices)

        assert out.value.shape.dims == (1, 3, 2)
        assert out.value.data == [9.0, 9.0, 1.0, 2.0, 5.0, 6.0]

    def test_embedding_padding_idx_does_not_accumulate_grad(self):
        """测试 padding_idx 对应的 embedding 不接收梯度。"""
        from tinytorch.nn.layers import Embedding

        emb = Embedding(num_embeddings=4, embedding_dim=2, padding_idx=0)
        emb.weight.value = NdArray([[9.0, 9.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        indices = Tensor(NdArray([[0, 1, 0, 2]]), requires_grad=False)
        out = emb(indices)
        loss = out.sum()
        loss.backward()

        assert emb.weight.grad is not None
        assert emb.weight.grad.data == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]


class TestAttention:
    """注意力层的测试。"""
    
    def test_attention_creation(self):
        """测试注意力层创建。"""
        try:
            from tinytorch.nn.layers import MultiHeadAttention
            attn = MultiHeadAttention(embed_dim=512, num_heads=8)
            assert attn.embed_dim == 512
            assert attn.num_heads == 8
        except ImportError:
            pytest.skip("MultiHeadAttention 暂未实现")

    def test_attention_forward_matches_expected_values(self):
        """测试固定投影下的注意力前向结果。"""
        from tinytorch.nn.layers import MultiHeadAttention

        attn = MultiHeadAttention(embed_dim=2, num_heads=1, dropout=0.0)
        for proj in [attn.W_q, attn.W_k, attn.W_v, attn.W_o]:
            proj.weight.value = NdArray([[1.0, 0.0], [0.0, 1.0]])
            proj.bias.value = NdArray([0.0, 0.0])

        x = Tensor(NdArray([[[1.0, 0.0], [0.0, 1.0]]]), requires_grad=False)
        y = attn(x)

        score = math.exp(1.0 / math.sqrt(2.0))
        keep_self = score / (score + 1.0)
        mix_other = 1.0 / (score + 1.0)
        expected = [keep_self, mix_other, mix_other, keep_self]

        assert y.value.shape.dims == (1, 2, 2)
        for actual, target in zip(y.value.data, expected):
            assert abs(actual - target) < 1e-9

    def test_attention_backward_matches_expected_gradients(self):
        """测试固定投影下的注意力反向梯度。"""
        from tinytorch.nn.layers import MultiHeadAttention

        attn = MultiHeadAttention(embed_dim=2, num_heads=1, dropout=0.0)
        for proj in [attn.W_q, attn.W_k, attn.W_v, attn.W_o]:
            proj.weight.value = NdArray([[1.0, 0.0], [0.0, 1.0]])
            proj.bias.value = NdArray([0.0, 0.0])

        x = Tensor(NdArray([[[1.0, 0.0], [0.0, 1.0]]]))
        y = attn(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.data == [1.0, 1.0, 1.0, 1.0]
        assert attn.W_q.weight.grad.data == [0.0, 0.0, 0.0, 0.0]
        assert attn.W_k.weight.grad.data == [0.0, 0.0, 0.0, 0.0]
        assert attn.W_v.weight.grad.data == [1.0, 1.0, 1.0, 1.0]
        assert attn.W_o.weight.grad.data == [1.0, 1.0, 1.0, 1.0]
        assert attn.W_q.bias.grad.data == [0.0, 0.0]
        assert attn.W_k.bias.grad.data == [0.0, 0.0]
        assert attn.W_v.bias.grad.data == [2.0, 2.0]
        assert attn.W_o.bias.grad.data == [2.0, 2.0]


class TestRNNLayers:
    """循环层基础测试。"""

    def test_gru_input_size_validation(self):
        """测试 GRU 会校验输入特征维度。"""
        try:
            from tinytorch.nn.layers import GRU
            gru = GRU(input_size=3, hidden_size=4)
            x = Tensor(NdArray.randn((2, 5, 2)))
            with pytest.raises(ValueError):
                _ = gru(x)
        except ImportError:
            pytest.skip("GRU 暂未实现")

    def test_rnn_backward(self):
        """测试 RNN 参数可反向传播。"""
        try:
            from tinytorch.nn.layers import RNN
            rnn = RNN(input_size=3, hidden_size=4)
            x = Tensor(NdArray.randn((2, 5, 3)))
            y = rnn(x)
            loss = y.sum()
            loss.backward()
            assert x.grad is not None
            assert rnn.W_ih.grad is not None
            assert rnn.W_hh.grad is not None
        except ImportError:
            pytest.skip("RNN 暂未实现")

    def test_rnn_forward_matches_expected_values(self):
        """测试 RNN 在固定权重下的前向结果。"""
        from tinytorch.nn.layers import RNN

        rnn = RNN(input_size=1, hidden_size=1)
        rnn.W_ih.value = NdArray([[1.0]])
        rnn.W_hh.value = NdArray([[1.0]])
        rnn.bias.value = NdArray([0.0])

        x = Tensor(NdArray([[[1.0], [1.0]]]), requires_grad=False)
        y = rnn(x)

        expected = [math.tanh(1.0), math.tanh(1.0 + math.tanh(1.0))]
        for actual, target in zip(y.value.data, expected):
            assert abs(actual - target) < 1e-9

    def test_lstm_backward(self):
        """测试 LSTM 参数可反向传播。"""
        try:
            from tinytorch.nn.layers import LSTM
            lstm = LSTM(input_size=3, hidden_size=4)
            x = Tensor(NdArray.randn((2, 4, 3)))
            y, c = lstm(x)
            loss = y.sum() + c.sum()
            loss.backward()
            assert x.grad is not None
            assert lstm.W_ii.grad is not None
            assert lstm.W_hi.grad is not None
            assert lstm.W_io.grad is not None
        except ImportError:
            pytest.skip("LSTM 暂未实现")

    def test_lstm_forward_matches_expected_values(self):
        """测试 LSTM 在固定权重下的前向结果。"""
        from tinytorch.nn.layers import LSTM

        lstm = LSTM(input_size=1, hidden_size=1)
        lstm.W_ii.value = NdArray([[0.0]])
        lstm.W_hi.value = NdArray([[0.0]])
        lstm.W_if.value = NdArray([[0.0]])
        lstm.W_hf.value = NdArray([[0.0]])
        lstm.W_ig.value = NdArray([[1.0]])
        lstm.W_hg.value = NdArray([[0.0]])
        lstm.W_io.value = NdArray([[0.0]])
        lstm.W_ho.value = NdArray([[0.0]])
        lstm.b_i.value = NdArray([0.0])
        lstm.b_f.value = NdArray([0.0])
        lstm.b_g.value = NdArray([0.0])
        lstm.b_o.value = NdArray([0.0])

        x = Tensor(NdArray([[[1.0]]]), requires_grad=False)
        y, c = lstm(x)

        g = math.tanh(1.0)
        expected_c = 0.5 * g
        expected_h = 0.5 * math.tanh(expected_c)

        assert abs(y.value.data[0] - expected_h) < 1e-9
        assert abs(c.value.data[0] - expected_c) < 1e-9

    def test_gru_backward(self):
        """测试 GRU 参数可反向传播。"""
        try:
            from tinytorch.nn.layers import GRU
            gru = GRU(input_size=3, hidden_size=4)
            x = Tensor(NdArray.randn((2, 4, 3)))
            y = gru(x)
            loss = y.sum()
            loss.backward()
            assert x.grad is not None
            assert gru.W_ir.grad is not None
            assert gru.W_hn.grad is not None
        except ImportError:
            pytest.skip("GRU 暂未实现")

    def test_gru_forward_matches_expected_values(self):
        """测试 GRU 在固定权重下的前向结果。"""
        from tinytorch.nn.layers import GRU

        gru = GRU(input_size=1, hidden_size=1)
        gru.W_ir.value = NdArray([[0.0]])
        gru.W_hr.value = NdArray([[0.0]])
        gru.W_iz.value = NdArray([[0.0]])
        gru.W_hz.value = NdArray([[0.0]])
        gru.W_in.value = NdArray([[1.0]])
        gru.W_hn.value = NdArray([[0.0]])
        gru.b_r.value = NdArray([0.0])
        gru.b_z.value = NdArray([0.0])
        gru.b_n.value = NdArray([0.0])

        x = Tensor(NdArray([[[1.0]]]), requires_grad=False)
        y = gru(x)

        expected = 0.5 * math.tanh(1.0)
        assert abs(y.value.data[0] - expected) < 1e-9

    def test_attention_mask(self):
        """测试注意力 mask 生效。"""
        try:
            from tinytorch.nn.layers import MultiHeadAttention
            attn = MultiHeadAttention(embed_dim=4, num_heads=2, dropout=0.0)
            x = Tensor(NdArray([[[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]]), requires_grad=False)
            # 只允许看第一个 token
            mask = Tensor(NdArray([[[1.0, 0.0], [1.0, 0.0]]]), requires_grad=False)
            out_masked = attn(x, mask=mask)
            out_unmasked = attn(x)
            assert out_masked.value.shape.dims == out_unmasked.value.shape.dims
            assert out_masked.value.data != out_unmasked.value.data
        except ImportError:
            pytest.skip("MultiHeadAttention 暂未实现")

    def test_attention_backward_to_qkv(self):
        """测试注意力能反传到 Q/K/V 投影参数。"""
        try:
            from tinytorch.nn.layers import MultiHeadAttention
            attn = MultiHeadAttention(embed_dim=4, num_heads=2, dropout=0.0)
            x = Tensor(NdArray.randn((1, 3, 4)))
            y = attn(x)
            loss = y.sum()
            loss.backward()
            assert attn.W_q.weight.grad is not None
            assert attn.W_k.weight.grad is not None
            assert attn.W_v.weight.grad is not None
            assert attn.W_o.weight.grad is not None
        except ImportError:
            pytest.skip("MultiHeadAttention 暂未实现")

    def test_attention_cross_attention_backward(self):
        """测试 cross-attention（q_len != kv_len）可前后向传播。"""
        try:
            from tinytorch.nn.layers import MultiHeadAttention
            attn = MultiHeadAttention(embed_dim=4, num_heads=2, dropout=0.0)
            query = Tensor(NdArray.randn((1, 2, 4)))
            key = Tensor(NdArray.randn((1, 3, 4)))
            value = Tensor(NdArray.randn((1, 3, 4)))
            y = attn(query, key, value)
            assert y.value.shape.dims == (1, 2, 4)
            loss = y.sum()
            loss.backward()
            assert query.grad is not None
            assert key.grad is not None
            assert value.grad is not None
            assert attn.W_q.weight.grad is not None
            assert attn.W_k.weight.grad is not None
            assert attn.W_v.weight.grad is not None
        except ImportError:
            pytest.skip("MultiHeadAttention 暂未实现")


class TestModule:
    """模块基类的测试。"""
    
    def test_module_parameters(self):
        """测试获取模块参数。"""
        layer = Linear(10, 5)
        params = list(layer.parameters())
        assert len(params) > 0
        
    def test_module_train_mode(self):
        """测试训练模式。"""
        layer = Linear(10, 5)
        layer.train()
        assert layer.training == True
        
    def test_module_eval_mode(self):
        """测试评估模式。"""
        layer = Linear(10, 5)
        layer.eval()
        assert layer.training == False

    def test_tensor_attr_not_auto_buffer(self):
        """测试普通 Tensor 属性不会自动注册为 buffer。"""
        class Dummy(Module):
            def forward(self, x):
                return x

        m = Dummy()
        m.cache = Tensor(NdArray([1.0]), requires_grad=False)
        assert 'cache' not in m._buffers

    def test_parameter_replacement_unregisters_old_entry(self):
        """测试参数被普通值覆盖后会从参数注册表移除。"""
        class Dummy(Module):
            def forward(self, x):
                return x

        m = Dummy()
        m.p = Parameter(NdArray([1.0]))
        assert 'p' in m._parameters
        m.p = None
        assert 'p' not in m._parameters

    def test_module_deletion_unregisters_entry(self):
        """测试删除子模块时会同步更新模块注册表。"""
        class Dummy(Module):
            def forward(self, x):
                return x

        parent = Dummy()
        parent.child = Dummy()
        assert 'child' in parent._modules
        del parent.child
        assert 'child' not in parent._modules

    def test_state_dict_round_trip(self):
        """测试 state_dict/load_state_dict 可恢复参数和缓冲区。"""
        class Dummy(Module):
            def __init__(self):
                super().__init__()
                self.linear = Linear(2, 2)
                self.register_buffer('running_mean', Tensor(NdArray([1.0, 2.0]), requires_grad=False))

            def forward(self, x):
                return self.linear(x)

        source = Dummy()
        source.linear.weight.value = NdArray([[1.0, 2.0], [3.0, 4.0]])
        source.linear.bias.value = NdArray([5.0, 6.0])
        source.running_mean.value = NdArray([7.0, 8.0])

        target = Dummy()
        target.load_state_dict(source.state_dict())

        assert target.linear.weight.value.data == [1.0, 2.0, 3.0, 4.0]
        assert target.linear.bias.value.data == [5.0, 6.0]
        assert target.running_mean.value.data == [7.0, 8.0]

    def test_load_state_dict_strict_reports_missing_keys(self):
        """测试 strict=False 返回缺失键，strict=True 抛错。"""
        class Dummy(Module):
            def __init__(self):
                super().__init__()
                self.linear = Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        module = Dummy()
        state = module.state_dict()
        del state['linear.bias']

        result = module.load_state_dict(state, strict=False)
        assert result['missing_keys'] == ['linear.bias']
        assert result['unexpected_keys'] == []

        with pytest.raises(KeyError):
            module.load_state_dict(state, strict=True)


class TestSequential:
    """顺序容器的测试。"""
    
    def test_sequential_creation(self):
        """测试顺序容器创建。"""
        try:
            from tinytorch.nn import Sequential
            model = Sequential(
                Linear(10, 20),
                ReLU(),
                Linear(20, 5)
            )
            assert len(model.layers) == 3
        except ImportError:
            pytest.skip("Sequential 暂未实现")
        
    def test_sequential_forward(self):
        """测试顺序容器前向传播。"""
        try:
            from tinytorch.nn import Sequential
            model = Sequential(
                Linear(3, 5),
                ReLU(),
                Linear(5, 2)
            )
            x = Tensor(NdArray([[1.0, 2.0, 3.0]]))
            output = model(x)
            assert output.value.shape.dims[1] == 2
        except ImportError:
            pytest.skip("Sequential 暂未实现")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
