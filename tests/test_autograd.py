"""测试自动微分功能。

Author: TinyAI Team
"""

import pytest
from tinytorch.ndarr import NdArray, Shape
from tinytorch.autograd import Tensor, no_grad
from tinytorch.autograd.ops.basic import Add, Sub, Mul, Div, Neg
from tinytorch.autograd.ops.math_ops import Exp, Log, Pow, Sqrt
from tinytorch.autograd.ops.matrix import MatMul, Transpose, Reshape
from tinytorch.autograd.ops.reduce import Sum, Mean
from tinytorch.autograd.ops.activation import ReLU, Sigmoid, Tanh


class TestVariable:
    """Tensor 类的测试。"""
    
    def test_variable_creation(self):
        """测试变量创建。"""
        t = NdArray([1.0, 2.0, 3.0])
        v = Tensor(t, name="test_var")
        assert v.name == "test_var"
        assert v.requires_grad == True
        assert v.grad is None
        
    def test_variable_no_grad(self):
        """测试不需要梯度的变量。"""
        t = NdArray([1.0, 2.0, 3.0])
        v = Tensor(t, requires_grad=False)
        assert v.requires_grad == False
        
    def test_variable_detach(self):
        """测试变量分离。"""
        t = NdArray([1.0, 2.0, 3.0])
        v = Tensor(t, name="test")
        detached = v.detach()
        assert detached.requires_grad == False
        assert "detached" in detached.name

    def test_no_grad_context(self):
        """测试 no_grad 上下文。"""
        x = Tensor(NdArray([1.0, 2.0, 3.0]))
        with no_grad():
            y = x * 2
        assert y.requires_grad == False
        assert y.creator is None


class TestBasicOps:
    """基本运算的测试。"""
    
    def test_add_forward(self):
        """测试加法前向传播。"""
        x = Tensor(NdArray([1.0, 2.0, 3.0]))
        y = Tensor(NdArray([4.0, 5.0, 6.0]))
        z = x.add(y)
        assert z.value.data == [5.0, 7.0, 9.0]
        
    def test_add_operator(self):
        """测试加法运算符重载。"""
        x = Tensor(NdArray([1.0, 2.0]))
        y = Tensor(NdArray([3.0, 4.0]))
        z = x + y
        assert z.value.data == [4.0, 6.0]
        
    def test_sub_forward(self):
        """测试减法前向传播。"""
        x = Tensor(NdArray([5.0, 7.0, 9.0]))
        y = Tensor(NdArray([1.0, 2.0, 3.0]))
        z = x.sub(y)
        assert z.value.data == [4.0, 5.0, 6.0]
        
    def test_mul_forward(self):
        """测试乘法前向传播。"""
        x = Tensor(NdArray([2.0, 3.0, 4.0]))
        y = Tensor(NdArray([5.0, 6.0, 7.0]))
        z = x.mul(y)
        assert z.value.data == [10.0, 18.0, 28.0]
        
    def test_div_forward(self):
        """测试除法前向传播。"""
        x = Tensor(NdArray([10.0, 20.0, 30.0]))
        y = Tensor(NdArray([2.0, 4.0, 5.0]))
        z = x.div(y)
        assert z.value.data == [5.0, 5.0, 6.0]
        
    def test_neg_forward(self):
        """测试取负前向传播。"""
        x = Tensor(NdArray([1.0, -2.0, 3.0]))
        z = x.neg()
        assert z.value.data == [-1.0, 2.0, -3.0]
        
    def test_neg_operator(self):
        """测试取负运算符重载。"""
        x = Tensor(NdArray([1.0, -2.0]))
        z = -x
        assert z.value.data == [-1.0, 2.0]


class TestMathOps:
    """数学运算的测试。"""
    
    def test_exp_forward(self):
        """测试指数运算。"""
        x = Tensor(NdArray([0.0, 1.0, 2.0]))
        z = x.exp()
        # e^0 = 1, e^1 ≈ 2.718, e^2 ≈ 7.389
        assert abs(z.value.data[0] - 1.0) < 0.01
        assert abs(z.value.data[1] - 2.718) < 0.01

    def test_exp_extreme_values(self):
        """测试极值输入下 exp 不抛出异常。"""
        x = Tensor(NdArray([1000.0, -1000.0]))
        z = x.exp()
        assert z.value.data[0] == float('inf')
        assert z.value.data[1] == 0.0
        
    def test_log_forward(self):
        """测试对数运算。"""
        x = Tensor(NdArray([1.0, 2.718, 7.389]))
        z = x.log()
        # ln(1) = 0, ln(e) = 1
        assert abs(z.value.data[0] - 0.0) < 0.01
        
    def test_pow_forward(self):
        """测试幂运算。"""
        x = Tensor(NdArray([2.0, 3.0, 4.0]))
        z = x.pow(2)
        assert z.value.data == [4.0, 9.0, 16.0]
        
    def test_sqrt_forward(self):
        """测试平方根运算。"""
        x = Tensor(NdArray([4.0, 9.0, 16.0]))
        z = x.sqrt()
        assert z.value.data == [2.0, 3.0, 4.0]


class TestMatrixOps:
    """矩阵运算的测试。"""
    
    def test_matmul_forward(self):
        """测试矩阵乘法。"""
        x = Tensor(NdArray([[1.0, 2.0], [3.0, 4.0]]))
        y = Tensor(NdArray([[5.0, 6.0], [7.0, 8.0]]))
        z = x.matmul(y)
        # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert z.value.shape.dims == (2, 2)
        assert z.value.data == [19.0, 22.0, 43.0, 50.0]
        
    def test_transpose_forward(self):
        """测试转置运算。"""
        x = Tensor(NdArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        z = x.transpose()
        assert z.value.shape.dims == (3, 2)
        
    def test_reshape_forward(self):
        """测试重塑运算。"""
        x = Tensor(NdArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        z = x.reshape((2, 3))
        assert z.value.shape.dims == (2, 3)
        assert z.value.data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_matmul_backward(self):
        """测试矩阵乘法反向传播。"""
        x = Tensor(NdArray([[1.0, 2.0], [3.0, 4.0]]), name="x")
        y = Tensor(NdArray([[5.0, 6.0], [7.0, 8.0]]), name="y")
        z = x.matmul(y)
        z.backward(NdArray([[1.0, 1.0], [1.0, 1.0]]))
        # dL/dx = grad_output @ y.T, dL/dy = x.T @ grad_output
        assert x.grad is not None
        assert y.grad is not None
        assert x.grad.shape.dims == (2, 2)
        assert y.grad.shape.dims == (2, 2)
        # grad_output @ y.T: [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
        assert abs(x.grad.data[0] - 11.0) < 1e-6
        assert abs(x.grad.data[1] - 15.0) < 1e-6

    def test_transpose_backward(self):
        """测试转置反向传播。"""
        x = Tensor(NdArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), name="x")
        z = x.transpose()
        z.backward(NdArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        assert x.grad is not None
        # 转置的转置是原矩阵，梯度应转置回
        assert x.grad.shape.dims == (2, 3)
        assert x.grad.data == [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]

    def test_reshape_backward(self):
        """测试重塑反向传播。"""
        x = Tensor(NdArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), name="x")
        z = x.reshape((2, 3))
        z.backward(NdArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        assert x.grad is not None
        assert x.grad.data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_exp_backward(self):
        """测试指数反向传播。d(exp(x))/dx = exp(x)"""
        x = Tensor(NdArray([1.0, 2.0]), name="x")
        z = x.exp()
        z.backward(NdArray([1.0, 1.0]))
        assert x.grad is not None
        import math
        assert abs(x.grad.data[0] - math.e) < 1e-6
        assert abs(x.grad.data[1] - math.e ** 2) < 1e-5

    def test_log_backward(self):
        """测试对数反向传播。d(log(x))/dx = 1/x"""
        x = Tensor(NdArray([2.0, 3.0]), name="x")
        z = x.log()
        z.backward(NdArray([1.0, 1.0]))
        assert x.grad is not None
        assert abs(x.grad.data[0] - 0.5) < 1e-6
        assert abs(x.grad.data[1] - 1.0/3.0) < 1e-6

    def test_sqrt_backward(self):
        """测试平方根反向传播。d(sqrt(x))/dx = 1/(2*sqrt(x))"""
        x = Tensor(NdArray([4.0, 9.0]), name="x")
        z = x.sqrt()
        z.backward(NdArray([1.0, 1.0]))
        assert x.grad is not None
        assert abs(x.grad.data[0] - 0.25) < 1e-6  # 1/(2*2)=0.25
        assert abs(x.grad.data[1] - 1.0/6.0) < 1e-6  # 1/(2*3)

    def test_pow_backward(self):
        """测试幂运算反向传播。d(x^2)/dx = 2x"""
        x = Tensor(NdArray([3.0, 4.0]), name="x")
        z = x.pow(2)
        z.backward(NdArray([1.0, 1.0]))
        assert x.grad is not None
        assert abs(x.grad.data[0] - 6.0) < 1e-6  # 2*3
        assert abs(x.grad.data[1] - 8.0) < 1e-6  # 2*4


class TestReduceOps:
    """归约运算的测试。"""
    
    def test_sum_forward(self):
        """测试求和运算。"""
        x = Tensor(NdArray([1.0, 2.0, 3.0, 4.0]))
        z = x.sum()
        assert z.value.data[0] == 10.0
        
    def test_mean_forward(self):
        """测试求均值运算。"""
        x = Tensor(NdArray([2.0, 4.0, 6.0, 8.0]))
        z = x.mean()
        assert z.value.data[0] == 5.0

    def test_sum_backward_with_axis(self):
        """测试 sum(axis=0) 的反向传播。"""
        x = Tensor(NdArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        y = x.sum(axis=0)

        y.backward(NdArray([1.0, 2.0, 3.0]))

        assert x.grad is not None
        assert x.grad.data == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]

    def test_mean_backward_with_axis(self):
        """测试 mean(axis=1) 的反向传播。"""
        x = Tensor(NdArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        y = x.mean(axis=1)

        y.backward(NdArray([3.0, 6.0]))

        assert x.grad is not None
        assert x.grad.data == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]


class TestActivationOps:
    """激活函数的测试。"""
    
    def test_relu_forward(self):
        """测试 ReLU 激活。"""
        x = Tensor(NdArray([-2.0, -1.0, 0.0, 1.0, 2.0]))
        z = x.relu()
        assert z.value.data == [0.0, 0.0, 0.0, 1.0, 2.0]
        
    def test_sigmoid_forward(self):
        """测试 Sigmoid 激活。"""
        x = Tensor(NdArray([0.0]))
        z = x.sigmoid()
        # sigmoid(0) = 0.5
        assert abs(z.value.data[0] - 0.5) < 0.01

    def test_sigmoid_extreme_values(self):
        """测试 Sigmoid 在极值输入下保持数值稳定。"""
        x = Tensor(NdArray([1000.0, -1000.0]))
        z = x.sigmoid()
        assert abs(z.value.data[0] - 1.0) < 1e-9
        assert abs(z.value.data[1] - 0.0) < 1e-9
        
    def test_tanh_forward(self):
        """测试 Tanh 激活。"""
        x = Tensor(NdArray([0.0]))
        z = x.tanh()
        # tanh(0) = 0
        assert abs(z.value.data[0] - 0.0) < 0.01


class TestBackward:
    """反向传播的测试。"""
    
    def test_simple_backward(self):
        """测试简单反向传播。"""
        x = Tensor(NdArray([2.0]), name="x")
        y = Tensor(NdArray([3.0]), name="y")
        z = x + y
        z.backward()
        # dz/dx = 1, dz/dy = 1
        assert x.grad is not None
        assert y.grad is not None
        assert abs(x.grad.data[0] - 1.0) < 1e-9
        assert abs(y.grad.data[0] - 1.0) < 1e-9
        
    def test_mul_backward(self):
        """测试乘法反向传播。"""
        x = Tensor(NdArray([2.0]), name="x")
        y = Tensor(NdArray([3.0]), name="y")
        z = x * y
        z.backward()
        # dz/dx = y = 3, dz/dy = x = 2
        assert x.grad is not None
        assert y.grad is not None
        assert abs(x.grad.data[0] - 3.0) < 1e-9
        assert abs(y.grad.data[0] - 2.0) < 1e-9
        
    def test_chain_backward(self):
        """测试链式反向传播。"""
        x = Tensor(NdArray([2.0]), name="x")
        y = x * x  # y = x^2
        z = y + x  # z = x^2 + x
        z.backward()
        # dz/dx = 2x + 1 = 5
        assert x.grad is not None
        assert abs(x.grad.data[0] - 5.0) < 1e-9

    def test_non_scalar_backward_requires_grad_output(self):
        """测试非标量输出必须显式提供梯度。"""
        x = Tensor(NdArray([1.0, 2.0]), name="x")
        y = x * 2.0
        with pytest.raises(ValueError):
            y.backward()

    def test_non_scalar_backward_with_explicit_grad_output(self):
        """测试非标量输出传入梯度后可正常反传。"""
        x = Tensor(NdArray([1.0, 2.0]), name="x")
        y = x * 2.0
        y.backward(NdArray([3.0, 4.0]))
        # y = 2x, dy/dx = 2 => grad = [6, 8]
        assert x.grad is not None
        assert abs(x.grad.data[0] - 6.0) < 1e-9
        assert abs(x.grad.data[1] - 8.0) < 1e-9

    def test_broadcast_add_backward_sums_bias_gradient(self):
        """测试广播加法会沿广播维度累积梯度。"""
        x = Tensor(NdArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), name="x")
        bias = Tensor(NdArray([10.0, 20.0, 30.0]), name="bias")
        y = (x + bias).sum()

        y.backward()

        assert x.grad is not None
        assert bias.grad is not None
        assert x.grad.data == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert bias.grad.data == [2.0, 2.0, 2.0]

    def test_retain_graph_allows_second_backward(self):
        """测试 retain_graph=True 会保留计算图供第二次反传。"""
        x = Tensor(NdArray([2.0]), name="x")
        y = x * x

        y.backward(retain_graph=True)
        assert x.grad is not None
        assert x.grad.data == [4.0]
        assert y.creator is not None

        y.backward()
        assert x.grad.data == [8.0]
        assert y.creator is None
        
    def test_clear_grad(self):
        """测试清除梯度。"""
        x = Tensor(NdArray([2.0]), name="x")
        y = x * x
        y.backward()
        assert x.grad is not None
        x.clear_grad()
        assert x.grad is None


class TestComputationGraph:
    """计算图的测试。"""
    
    def test_graph_creation(self):
        """测试计算图创建。"""
        x = Tensor(NdArray([1.0]), name="x")
        y = Tensor(NdArray([2.0]), name="y")
        z = x + y
        assert z.creator is not None
        assert len(z.creator.inputs) == 2
        
    def test_graph_unchain(self):
        """测试计算图解链。"""
        x = Tensor(NdArray([1.0]), name="x")
        y = x * x
        assert y.creator is not None
        y.unchain_backward()
        assert y.creator is None


class TestScalarOperations:
    """标量运算的测试。"""
    
    def test_add_scalar(self):
        """测试与标量相加。"""
        x = Tensor(NdArray([1.0, 2.0, 3.0]))
        z = x + 5
        assert z.value.data == [6.0, 7.0, 8.0]

    def test_radd_scalar(self):
        """测试标量在左侧的加法（5 + x）。"""
        x = Tensor(NdArray([1.0, 2.0, 3.0]))
        z = 5 + x
        assert z.value.data == [6.0, 7.0, 8.0]

    def test_mul_scalar(self):
        """测试与标量相乘。"""
        x = Tensor(NdArray([1.0, 2.0, 3.0]))
        z = x * 2
        assert z.value.data == [2.0, 4.0, 6.0]

    def test_rmul_scalar(self):
        """测试标量在左侧的乘法（2 * x）。"""
        x = Tensor(NdArray([1.0, 2.0, 3.0]))
        z = 2 * x
        assert z.value.data == [2.0, 4.0, 6.0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
