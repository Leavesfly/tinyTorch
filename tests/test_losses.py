"""测试损失函数。

Author: TinyAI Team
"""

import pytest
from tinytorch.tensor import Tensor
from tinytorch.autograd import Variable
from tinytorch.ml.losses import MSELoss, CrossEntropyLoss


class TestMSELoss:
    """均方误差损失的测试。"""
    
    def test_mse_loss_creation(self):
        """测试 MSE 损失创建。"""
        loss_fn = MSELoss()
        assert loss_fn is not None
        
    def test_mse_loss_forward(self):
        """测试 MSE 损失前向传播。"""
        loss_fn = MSELoss()
        pred = Variable(Tensor([1.0, 2.0, 3.0]))
        target = Variable(Tensor([1.0, 2.0, 3.0]))
        loss = loss_fn(pred, target)
        # 预测完全正确，损失应该接近0
        assert abs(loss.value.data[0]) < 0.01
        
    def test_mse_loss_with_error(self):
        """测试有误差的 MSE 损失。"""
        loss_fn = MSELoss()
        pred = Variable(Tensor([1.0, 2.0, 3.0]))
        target = Variable(Tensor([2.0, 3.0, 4.0]))
        loss = loss_fn(pred, target)
        # 每个元素误差为1，MSE = (1^2 + 1^2 + 1^2) / 3 = 1
        assert loss.value.data[0] > 0


class TestCrossEntropyLoss:
    """交叉熵损失的测试。"""
    
    def test_cross_entropy_creation(self):
        """测试交叉熵损失创建。"""
        loss_fn = CrossEntropyLoss()
        assert loss_fn is not None
        
    def test_cross_entropy_forward(self):
        """测试交叉熵损失前向传播。"""
        loss_fn = CrossEntropyLoss()
        # logits: [batch_size, num_classes]
        logits = Variable(Tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]]))
        # targets: [batch_size] (类别索引)
        targets = Variable(Tensor([0.0, 1.0]))
        loss = loss_fn(logits, targets)
        assert loss.value.data[0] > 0


class TestLossProperties:
    """损失函数属性的测试。"""
    
    def test_loss_reduction_none(self):
        """测试不进行归约的损失。"""
        loss_fn = MSELoss(reduction='none')
        pred = Variable(Tensor([1.0, 2.0, 3.0]))
        target = Variable(Tensor([2.0, 3.0, 4.0]))
        loss = loss_fn(pred, target)
        # 应该返回每个样本的损失
        assert loss.value.shape.size == 3
        
    def test_loss_reduction_mean(self):
        """测试均值归约的损失。"""
        loss_fn = MSELoss(reduction='mean')
        pred = Variable(Tensor([1.0, 2.0, 3.0]))
        target = Variable(Tensor([2.0, 3.0, 4.0]))
        loss = loss_fn(pred, target)
        # 应该返回标量
        assert loss.value.shape.size == 1
        
    def test_loss_reduction_sum(self):
        """测试求和归约的损失。"""
        loss_fn = MSELoss(reduction='sum')
        pred = Variable(Tensor([1.0, 2.0, 3.0]))
        target = Variable(Tensor([2.0, 3.0, 4.0]))
        loss = loss_fn(pred, target)
        # 应该返回标量
        assert loss.value.shape.size == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
