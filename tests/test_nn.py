"""测试神经网络层。

Author: TinyAI Team
"""

import pytest
from tinytorch.nn.layers import Linear, ReLU, LayerNorm, Dropout
from tinytorch.nn.parameter import Parameter
from tinytorch.autograd import Variable
from tinytorch.tensor import Tensor


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
        x = Variable(Tensor([[1.0, 2.0, 3.0]]))
        output = layer(x)
        assert output.value.shape.dims[0] == 1
        assert output.value.shape.dims[1] == 2


class TestActivation:
    """激活函数的测试。"""
    
    def test_relu(self):
        """测试 ReLU 激活。"""
        relu = ReLU()
        x = Variable(Tensor([-1.0, 0.0, 1.0, 2.0]))
        output = relu(x)
        # ReLU 应该将负数变为0
        assert output.value.data[0] == 0.0
        assert output.value.data[3] == 2.0


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
            x = Variable(Tensor([[1.0, 2.0, 3.0]]))
            output = model(x)
            assert output.value.shape.dims[1] == 2
        except ImportError:
            pytest.skip("Sequential 暂未实现")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
