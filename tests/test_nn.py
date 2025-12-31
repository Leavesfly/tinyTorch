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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
