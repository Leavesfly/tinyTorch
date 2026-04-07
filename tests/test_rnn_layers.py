"""测试 RNN、LSTM、GRU 循环层以及 Sequential、ModuleList 容器。

Author: TinyAI Team
"""

import pytest
from tinytorch.nn.layers.rnn import RNN, LSTM, GRU
from tinytorch.nn.layers import Linear, ReLU
from tinytorch.nn.container import Sequential, ModuleList
from tinytorch.nn.module import Module
from tinytorch.nn.parameter import Parameter
from tinytorch.autograd import Tensor
from tinytorch.ndarr import NdArray
from tinytorch.utils import random as tt_random


class TestRNN:
    """RNN 层的测试。"""
    
    def test_rnn_creation(self):
        """测试 RNN 创建和属性验证。"""
        rnn = RNN(input_size=10, hidden_size=20)
        assert rnn.input_size == 10
        assert rnn.hidden_size == 20
        assert rnn.use_bias is True
        assert rnn.W_ih is not None
        assert rnn.W_hh is not None
        assert rnn.bias is not None
        
        # 验证权重形状
        assert rnn.W_ih.value.shape.dims == (20, 10)
        assert rnn.W_hh.value.shape.dims == (20, 20)
        assert rnn.bias.value.shape.dims == (20,)
    
    def test_rnn_forward(self):
        """测试 RNN 前向传播和输出形状验证。"""
        tt_random.seed(42)
        rnn = RNN(input_size=10, hidden_size=20)
        
        # 输入形状: (batch=2, seq_len=3, input_size=10)
        x = Tensor(NdArray.randn((2, 3, 10)))
        output = rnn(x)
        
        # 验证输出形状: (batch=2, seq_len=3, hidden_size=20)
        assert output.value.shape.dims == (2, 3, 20)
    
    def test_rnn_backward(self):
        """测试 RNN 梯度可以回传。"""
        tt_random.seed(42)
        rnn = RNN(input_size=10, hidden_size=20)
        
        # 输入形状: (batch=2, seq_len=3, input_size=10)
        x = Tensor(NdArray.randn((2, 3, 10)))
        output = rnn(x)
        
        # 计算损失并反向传播
        loss = output.sum()
        loss.backward()
        
        # 验证梯度存在
        assert x.grad is not None
        assert rnn.W_ih.grad is not None
        assert rnn.W_hh.grad is not None
        if rnn.use_bias:
            assert rnn.bias.grad is not None


class TestLSTM:
    """LSTM 层的测试。"""
    
    def test_lstm_creation(self):
        """测试 LSTM 创建和属性验证。"""
        lstm = LSTM(input_size=10, hidden_size=20)
        assert lstm.input_size == 10
        assert lstm.hidden_size == 20
        assert lstm.use_bias is True
        
        # 验证权重存在
        assert lstm.W_ii is not None  # 输入门
        assert lstm.W_hi is not None
        assert lstm.W_if is not None  # 遗忘门
        assert lstm.W_hf is not None
        assert lstm.W_ig is not None  # 候选单元状态
        assert lstm.W_hg is not None
        assert lstm.W_io is not None  # 输出门
        assert lstm.W_ho is not None
        
        # 验证偏置存在
        assert lstm.b_i is not None
        assert lstm.b_f is not None
        assert lstm.b_g is not None
        assert lstm.b_o is not None
    
    def test_lstm_forward(self):
        """测试 LSTM 前向传播和输出形状验证。"""
        tt_random.seed(42)
        lstm = LSTM(input_size=10, hidden_size=20)
        
        # 输入形状: (batch=2, seq_len=3, input_size=10)
        x = Tensor(NdArray.randn((2, 3, 10)))
        h, c = lstm(x)
        
        # 验证隐藏状态输出形状: (batch=2, seq_len=3, hidden_size=20)
        assert h.value.shape.dims == (2, 3, 20)
        
        # 验证单元状态输出形状: (batch=2, hidden_size=20)
        assert c.value.shape.dims == (2, 20)
    
    def test_lstm_backward(self):
        """测试 LSTM 梯度可以回传。"""
        tt_random.seed(42)
        lstm = LSTM(input_size=10, hidden_size=20)
        
        # 输入形状: (batch=2, seq_len=3, input_size=10)
        x = Tensor(NdArray.randn((2, 3, 10)))
        h, c = lstm(x)
        
        # 计算损失并反向传播
        loss = h.sum() + c.sum()
        loss.backward()
        
        # 验证梯度存在
        assert x.grad is not None
        assert lstm.W_ii.grad is not None
        assert lstm.W_hi.grad is not None
        assert lstm.W_if.grad is not None
        assert lstm.W_hf.grad is not None
        assert lstm.W_ig.grad is not None
        assert lstm.W_hg.grad is not None
        assert lstm.W_io.grad is not None
        assert lstm.W_ho.grad is not None
        if lstm.use_bias:
            assert lstm.b_i.grad is not None
            assert lstm.b_f.grad is not None
            assert lstm.b_g.grad is not None
            assert lstm.b_o.grad is not None


class TestGRU:
    """GRU 层的测试。"""
    
    def test_gru_creation(self):
        """测试 GRU 创建和属性验证。"""
        gru = GRU(input_size=10, hidden_size=20)
        assert gru.input_size == 10
        assert gru.hidden_size == 20
        assert gru.use_bias is True
        
        # 验证权重存在
        assert gru.W_ir is not None  # 重置门
        assert gru.W_hr is not None
        assert gru.W_iz is not None  # 更新门
        assert gru.W_hz is not None
        assert gru.W_in is not None  # 新候选状态
        assert gru.W_hn is not None
        
        # 验证偏置存在
        assert gru.b_r is not None
        assert gru.b_z is not None
        assert gru.b_n is not None
    
    def test_gru_forward(self):
        """测试 GRU 前向传播和输出形状验证。"""
        tt_random.seed(42)
        gru = GRU(input_size=10, hidden_size=20)
        
        # 输入形状: (batch=2, seq_len=3, input_size=10)
        x = Tensor(NdArray.randn((2, 3, 10)))
        output = gru(x)
        
        # 验证输出形状: (batch=2, seq_len=3, hidden_size=20)
        assert output.value.shape.dims == (2, 3, 20)
    
    def test_gru_backward(self):
        """测试 GRU 梯度可以回传。"""
        tt_random.seed(42)
        gru = GRU(input_size=10, hidden_size=20)
        
        # 输入形状: (batch=2, seq_len=3, input_size=10)
        x = Tensor(NdArray.randn((2, 3, 10)))
        output = gru(x)
        
        # 计算损失并反向传播
        loss = output.sum()
        loss.backward()
        
        # 验证梯度存在
        assert x.grad is not None
        assert gru.W_ir.grad is not None
        assert gru.W_hr.grad is not None
        assert gru.W_iz.grad is not None
        assert gru.W_hz.grad is not None
        assert gru.W_in.grad is not None
        assert gru.W_hn.grad is not None
        if gru.use_bias:
            assert gru.b_r.grad is not None
            assert gru.b_z.grad is not None
            assert gru.b_n.grad is not None


class TestSequential:
    """Sequential 容器的测试。"""
    
    def test_sequential_creation(self):
        """测试 Sequential 创建。"""
        seq = Sequential(
            Linear(3, 5),
            ReLU(),
            Linear(5, 2)
        )
        
        # 验证层数
        assert len(seq) == 3
        
        # 验证每一层的类型
        assert isinstance(seq[0], Linear)
        assert isinstance(seq[1], ReLU)
        assert isinstance(seq[2], Linear)
        
        # 验证层的参数
        assert seq[0].in_features == 3
        assert seq[0].out_features == 5
        assert seq[2].in_features == 5
        assert seq[2].out_features == 2
    
    def test_sequential_forward(self):
        """测试 Sequential 前向传播。"""
        tt_random.seed(42)
        seq = Sequential(
            Linear(3, 5),
            ReLU(),
            Linear(5, 2)
        )
        
        # 输入形状: (batch=4, features=3)
        x = Tensor(NdArray.randn((4, 3)))
        output = seq(x)
        
        # 验证输出形状: (batch=4, features=2)
        assert output.value.shape.dims == (4, 2)
    
    def test_sequential_parameters(self):
        """测试 Sequential parameters() 返回所有参数。"""
        tt_random.seed(42)
        seq = Sequential(
            Linear(3, 5),
            ReLU(),
            Linear(5, 2)
        )
        
        # 获取所有参数
        params = list(seq.parameters())
        
        # 验证参数数量: 两个 Linear 层，每个有 weight 和 bias
        assert len(params) == 4
        
        # 验证参数形状
        assert params[0].value.shape.dims == (5, 3)  # Linear(3, 5).weight
        assert params[1].value.shape.dims == (5,)     # Linear(3, 5).bias
        assert params[2].value.shape.dims == (2, 5)  # Linear(5, 2).weight
        assert params[3].value.shape.dims == (2,)     # Linear(5, 2).bias
    
    def test_sequential_repr(self):
        """测试 Sequential __repr__ 不报错。"""
        seq = Sequential(
            Linear(3, 5),
            ReLU(),
            Linear(5, 2)
        )
        
        # 验证 __repr__ 可以正常调用且包含关键信息
        repr_str = repr(seq)
        assert 'Sequential' in repr_str
        assert 'Linear' in repr_str
        assert 'ReLU' in repr_str


class TestModuleList:
    """ModuleList 容器的测试。"""
    
    def test_modulelist_creation(self):
        """测试 ModuleList 创建。"""
        tt_random.seed(42)
        module_list = ModuleList([
            Linear(3, 5),
            Linear(5, 2)
        ])
        
        # 验证层数
        assert len(module_list) == 2
        
        # 验证每一层的类型
        assert isinstance(module_list[0], Linear)
        assert isinstance(module_list[1], Linear)
        
        # 验证层的参数
        assert module_list[0].in_features == 3
        assert module_list[0].out_features == 5
        assert module_list[1].in_features == 5
        assert module_list[1].out_features == 2
    
    def test_modulelist_append(self):
        """测试 ModuleList append 方法。"""
        tt_random.seed(42)
        module_list = ModuleList()
        
        # 初始为空
        assert len(module_list) == 0
        
        # 添加第一个层
        linear1 = Linear(3, 5)
        module_list.append(linear1)
        assert len(module_list) == 1
        assert module_list[0] is linear1
        
        # 添加第二个层
        linear2 = Linear(5, 2)
        module_list.append(linear2)
        assert len(module_list) == 2
        assert module_list[1] is linear2
    
    def test_modulelist_len(self):
        """测试 ModuleList __len__。"""
        tt_random.seed(42)
        module_list = ModuleList([
            Linear(3, 5),
            Linear(5, 2)
        ])
        
        assert len(module_list) == 2
        
        # 添加更多层
        module_list.append(ReLU())
        assert len(module_list) == 3
    
    def test_modulelist_getitem(self):
        """测试 ModuleList __getitem__。"""
        tt_random.seed(42)
        linear1 = Linear(3, 5)
        linear2 = Linear(5, 2)
        relu = ReLU()
        
        module_list = ModuleList([linear1, linear2, relu])
        
        # 测试正向索引
        assert module_list[0] is linear1
        assert module_list[1] is linear2
        assert module_list[2] is relu
        
        # 测试反向索引
        assert module_list[-1] is relu
        assert module_list[-2] is linear2
        assert module_list[-3] is linear1
    
    def test_modulelist_parameters(self):
        """测试 ModuleList parameters() 返回所有参数。"""
        tt_random.seed(42)
        module_list = ModuleList([
            Linear(3, 5),
            Linear(5, 2)
        ])
        
        # 获取所有参数
        params = list(module_list.parameters())
        
        # 验证参数数量: 两个 Linear 层，每个有 weight 和 bias
        assert len(params) == 4
        
        # 验证参数形状
        assert params[0].value.shape.dims == (5, 3)  # Linear(3, 5).weight
        assert params[1].value.shape.dims == (5,)     # Linear(3, 5).bias
        assert params[2].value.shape.dims == (2, 5)  # Linear(5, 2).weight
        assert params[3].value.shape.dims == (2,)     # Linear(5, 2).bias
