"""测试优化器。

Author: TinyAI Team
"""

import pytest
from tinytorch.tensor import Tensor
from tinytorch.autograd import Variable
from tinytorch.nn.parameter import Parameter
from tinytorch.ml.optimizers import SGD, Adam


class TestSGD:
    """SGD 优化器的测试。"""
    
    def test_sgd_creation(self):
        """测试 SGD 优化器创建。"""
        params = [Parameter(Tensor([1.0, 2.0, 3.0]))]
        optimizer = SGD(params, lr=0.01)
        assert optimizer.lr == 0.01
        assert len(optimizer.params) == 1
        
    def test_sgd_with_momentum(self):
        """测试带动量的 SGD。"""
        params = [Parameter(Tensor([1.0, 2.0, 3.0]))]
        optimizer = SGD(params, lr=0.01, momentum=0.9)
        assert optimizer.momentum == 0.9
        
    def test_sgd_zero_grad(self):
        """测试梯度清零。"""
        params = [Parameter(Tensor([1.0, 2.0, 3.0]))]
        optimizer = SGD(params, lr=0.01)
        # 设置梯度
        params[0].grad = Tensor([0.1, 0.2, 0.3])
        # 清零
        optimizer.zero_grad()
        assert params[0].grad is None
        
    def test_sgd_step(self):
        """测试优化器更新步骤。"""
        params = [Parameter(Tensor([1.0, 2.0, 3.0]))]
        optimizer = SGD(params, lr=0.1)
        # 设置梯度
        params[0].grad = Tensor([0.1, 0.2, 0.3])
        # 记录原始值
        original_data = params[0].data.copy()
        # 执行更新
        optimizer.step()
        # 参数应该改变（向梯度反方向移动）
        # 新值 = 旧值 - lr * grad
        assert params[0].data[0] == original_data[0] - 0.1 * 0.1


class TestAdam:
    """Adam 优化器的测试。"""
    
    def test_adam_creation(self):
        """测试 Adam 优化器创建。"""
        params = [Parameter(Tensor([1.0, 2.0, 3.0]))]
        optimizer = Adam(params, lr=0.001)
        assert optimizer.lr == 0.001
        assert len(optimizer.params) == 1
        
    def test_adam_with_betas(self):
        """测试 Adam 的 beta 参数。"""
        params = [Parameter(Tensor([1.0, 2.0, 3.0]))]
        optimizer = Adam(params, lr=0.001, betas=(0.9, 0.999))
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.999
        
    def test_adam_zero_grad(self):
        """测试 Adam 梯度清零。"""
        params = [Parameter(Tensor([1.0, 2.0, 3.0]))]
        optimizer = Adam(params, lr=0.001)
        # 设置梯度
        params[0].grad = Tensor([0.1, 0.2, 0.3])
        # 清零
        optimizer.zero_grad()
        assert params[0].grad is None
        
    def test_adam_step(self):
        """测试 Adam 更新步骤。"""
        params = [Parameter(Tensor([1.0, 2.0, 3.0]))]
        optimizer = Adam(params, lr=0.01)
        # 设置梯度
        params[0].grad = Tensor([0.1, 0.2, 0.3])
        # 记录原始值
        original_data = params[0].data.copy()
        # 执行更新
        optimizer.step()
        # 参数应该改变
        assert params[0].data != original_data


class TestOptimizerInterface:
    """优化器接口的测试。"""
    
    def test_multiple_parameters(self):
        """测试多参数优化。"""
        param1 = Parameter(Tensor([1.0, 2.0]))
        param2 = Parameter(Tensor([3.0, 4.0]))
        optimizer = SGD([param1, param2], lr=0.01)
        assert len(optimizer.params) == 2
        
    def test_learning_rate_change(self):
        """测试学习率修改。"""
        params = [Parameter(Tensor([1.0, 2.0, 3.0]))]
        optimizer = SGD(params, lr=0.01)
        assert optimizer.lr == 0.01
        # 修改学习率
        optimizer.lr = 0.001
        assert optimizer.lr == 0.001
        
    def test_empty_parameters(self):
        """测试空参数列表。"""
        optimizer = SGD([], lr=0.01)
        assert len(optimizer.params) == 0
        # 应该可以正常调用但不做任何事
        optimizer.zero_grad()
        optimizer.step()


class TestOptimizerBehavior:
    """优化器行为的测试。"""
    
    def test_sgd_convergence_direction(self):
        """测试 SGD 收敛方向。"""
        # 模拟一个简单的优化问题：最小化 f(x) = x^2
        x = Parameter(Tensor([10.0]))
        optimizer = SGD([x], lr=0.1)
        
        # 设置梯度 df/dx = 2x = 20
        x.grad = Tensor([20.0])
        original_value = x.data[0]
        
        # 执行更新
        optimizer.step()
        
        # x 应该向 0 移动（梯度下降）
        assert abs(x.data[0]) < abs(original_value)
        
    def test_multiple_steps(self):
        """测试多步优化。"""
        params = [Parameter(Tensor([1.0]))]
        optimizer = SGD(params, lr=0.1)
        
        # 执行多次更新
        for _ in range(3):
            params[0].grad = Tensor([1.0])
            optimizer.step()
            optimizer.zero_grad()
        
        # 参数应该减小了 3 * 0.1 * 1.0 = 0.3
        assert abs(params[0].data[0] - 0.7) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
