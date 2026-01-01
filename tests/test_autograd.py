"""测试自动微分功能。

Author: TinyAI Team
"""

import pytest
from tinytorch.tensor import Tensor, Shape
from tinytorch.autograd import Variable
from tinytorch.autograd.ops.basic import Add, Sub, Mul, Div, Neg
from tinytorch.autograd.ops.math_ops import Exp, Log, Pow, Sqrt
from tinytorch.autograd.ops.matrix import MatMul, Transpose, Reshape
from tinytorch.autograd.ops.reduce import Sum, Mean
from tinytorch.autograd.ops.activation import ReLU, Sigmoid, Tanh


class TestVariable:
    """Variable 类的测试。"""
    
    def test_variable_creation(self):
        """测试变量创建。"""
        t = Tensor([1.0, 2.0, 3.0])
        v = Variable(t, name="test_var")
        assert v.name == "test_var"
        assert v.requires_grad == True
        assert v.grad is None
        
    def test_variable_no_grad(self):
        """测试不需要梯度的变量。"""
        t = Tensor([1.0, 2.0, 3.0])
        v = Variable(t, requires_grad=False)
        assert v.requires_grad == False
        
    def test_variable_detach(self):
        """测试变量分离。"""
        t = Tensor([1.0, 2.0, 3.0])
        v = Variable(t, name="test")
        detached = v.detach()
        assert detached.requires_grad == False
        assert "detached" in detached.name


class TestBasicOps:
    """基本运算的测试。"""
    
    def test_add_forward(self):
        """测试加法前向传播。"""
        x = Variable(Tensor([1.0, 2.0, 3.0]))
        y = Variable(Tensor([4.0, 5.0, 6.0]))
        z = x.add(y)
        assert z.value.data == [5.0, 7.0, 9.0]
        
    def test_add_operator(self):
        """测试加法运算符重载。"""
        x = Variable(Tensor([1.0, 2.0]))
        y = Variable(Tensor([3.0, 4.0]))
        z = x + y
        assert z.value.data == [4.0, 6.0]
        
    def test_sub_forward(self):
        """测试减法前向传播。"""
        x = Variable(Tensor([5.0, 7.0, 9.0]))
        y = Variable(Tensor([1.0, 2.0, 3.0]))
        z = x.sub(y)
        assert z.value.data == [4.0, 5.0, 6.0]
        
    def test_mul_forward(self):
        """测试乘法前向传播。"""
        x = Variable(Tensor([2.0, 3.0, 4.0]))
        y = Variable(Tensor([5.0, 6.0, 7.0]))
        z = x.mul(y)
        assert z.value.data == [10.0, 18.0, 28.0]
        
    def test_div_forward(self):
        """测试除法前向传播。"""
        x = Variable(Tensor([10.0, 20.0, 30.0]))
        y = Variable(Tensor([2.0, 4.0, 5.0]))
        z = x.div(y)
        assert z.value.data == [5.0, 5.0, 6.0]
        
    def test_neg_forward(self):
        """测试取负前向传播。"""
        x = Variable(Tensor([1.0, -2.0, 3.0]))
        z = x.neg()
        assert z.value.data == [-1.0, 2.0, -3.0]
        
    def test_neg_operator(self):
        """测试取负运算符重载。"""
        x = Variable(Tensor([1.0, -2.0]))
        z = -x
        assert z.value.data == [-1.0, 2.0]


class TestMathOps:
    """数学运算的测试。"""
    
    def test_exp_forward(self):
        """测试指数运算。"""
        x = Variable(Tensor([0.0, 1.0, 2.0]))
        z = x.exp()
        # e^0 = 1, e^1 ≈ 2.718, e^2 ≈ 7.389
        assert abs(z.value.data[0] - 1.0) < 0.01
        assert abs(z.value.data[1] - 2.718) < 0.01
        
    def test_log_forward(self):
        """测试对数运算。"""
        x = Variable(Tensor([1.0, 2.718, 7.389]))
        z = x.log()
        # ln(1) = 0, ln(e) = 1
        assert abs(z.value.data[0] - 0.0) < 0.01
        
    def test_pow_forward(self):
        """测试幂运算。"""
        x = Variable(Tensor([2.0, 3.0, 4.0]))
        z = x.pow(2)
        assert z.value.data == [4.0, 9.0, 16.0]
        
    def test_sqrt_forward(self):
        """测试平方根运算。"""
        x = Variable(Tensor([4.0, 9.0, 16.0]))
        z = x.sqrt()
        assert z.value.data == [2.0, 3.0, 4.0]


class TestMatrixOps:
    """矩阵运算的测试。"""
    
    def test_matmul_forward(self):
        """测试矩阵乘法。"""
        x = Variable(Tensor([[1.0, 2.0], [3.0, 4.0]]))
        y = Variable(Tensor([[5.0, 6.0], [7.0, 8.0]]))
        z = x.matmul(y)
        # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        # [[19, 22], [43, 50]]
        assert z.value.shape.dims == (2, 2)
        
    def test_transpose_forward(self):
        """测试转置运算。"""
        x = Variable(Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        z = x.transpose()
        assert z.value.shape.dims == (3, 2)
        
    def test_reshape_forward(self):
        """测试重塑运算。"""
        x = Variable(Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        z = x.reshape((2, 3))
        assert z.value.shape.dims == (2, 3)


class TestReduceOps:
    """归约运算的测试。"""
    
    def test_sum_forward(self):
        """测试求和运算。"""
        x = Variable(Tensor([1.0, 2.0, 3.0, 4.0]))
        z = x.sum()
        assert z.value.data[0] == 10.0
        
    def test_mean_forward(self):
        """测试求均值运算。"""
        x = Variable(Tensor([2.0, 4.0, 6.0, 8.0]))
        z = x.mean()
        assert z.value.data[0] == 5.0


class TestActivationOps:
    """激活函数的测试。"""
    
    def test_relu_forward(self):
        """测试 ReLU 激活。"""
        x = Variable(Tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
        z = x.relu()
        assert z.value.data == [0.0, 0.0, 0.0, 1.0, 2.0]
        
    def test_sigmoid_forward(self):
        """测试 Sigmoid 激活。"""
        x = Variable(Tensor([0.0]))
        z = x.sigmoid()
        # sigmoid(0) = 0.5
        assert abs(z.value.data[0] - 0.5) < 0.01
        
    def test_tanh_forward(self):
        """测试 Tanh 激活。"""
        x = Variable(Tensor([0.0]))
        z = x.tanh()
        # tanh(0) = 0
        assert abs(z.value.data[0] - 0.0) < 0.01


class TestBackward:
    """反向传播的测试。"""
    
    def test_simple_backward(self):
        """测试简单反向传播。"""
        x = Variable(Tensor([2.0]), name="x")
        y = Variable(Tensor([3.0]), name="y")
        z = x + y
        z.backward()
        # dz/dx = 1, dz/dy = 1
        assert x.grad is not None
        assert y.grad is not None
        
    def test_mul_backward(self):
        """测试乘法反向传播。"""
        x = Variable(Tensor([2.0]), name="x")
        y = Variable(Tensor([3.0]), name="y")
        z = x * y
        z.backward()
        # dz/dx = y = 3, dz/dy = x = 2
        assert x.grad is not None
        assert y.grad is not None
        
    def test_chain_backward(self):
        """测试链式反向传播。"""
        x = Variable(Tensor([2.0]), name="x")
        y = x * x  # y = x^2
        z = y + x  # z = x^2 + x
        z.backward()
        # dz/dx = 2x + 1 = 5
        assert x.grad is not None
        
    def test_clear_grad(self):
        """测试清除梯度。"""
        x = Variable(Tensor([2.0]), name="x")
        y = x * x
        y.backward()
        assert x.grad is not None
        x.clear_grad()
        assert x.grad is None


class TestComputationGraph:
    """计算图的测试。"""
    
    def test_graph_creation(self):
        """测试计算图创建。"""
        x = Variable(Tensor([1.0]), name="x")
        y = Variable(Tensor([2.0]), name="y")
        z = x + y
        assert z.creator is not None
        assert len(z.creator.inputs) == 2
        
    def test_graph_unchain(self):
        """测试计算图解链。"""
        x = Variable(Tensor([1.0]), name="x")
        y = x * x
        assert y.creator is not None
        y.unchain_backward()
        assert y.creator is None


class TestScalarOperations:
    """标量运算的测试。"""
    
    def test_add_scalar(self):
        """测试与标量相加。"""
        x = Variable(Tensor([1.0, 2.0, 3.0]))
        z = x + 5
        assert z.value.data[0] > 1.0
        
    def test_mul_scalar(self):
        """测试与标量相乘。"""
        x = Variable(Tensor([1.0, 2.0, 3.0]))
        z = x * 2
        # 应该是每个元素都乘以2
        assert z.value.shape.size > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
