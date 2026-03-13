"""测试损失函数。

Author: TinyAI Team
"""

import math

import pytest
from tinytorch.ndarr import NdArray
from tinytorch.autograd import Tensor
from tinytorch.ml.losses import MSELoss, CrossEntropyLoss, BCELoss


class TestMSELoss:
    """均方误差损失的测试。"""
    
    def test_mse_loss_creation(self):
        """测试 MSE 损失创建。"""
        loss_fn = MSELoss()
        assert loss_fn is not None
        
    def test_mse_loss_forward(self):
        """测试 MSE 损失前向传播。"""
        loss_fn = MSELoss()
        pred = Tensor(NdArray([1.0, 2.0, 3.0]))
        target = Tensor(NdArray([1.0, 2.0, 3.0]))
        loss = loss_fn(pred, target)
        # 预测完全正确，损失应该接近0
        assert abs(loss.value.data[0]) < 0.01
        
    def test_mse_loss_with_error(self):
        """测试有误差的 MSE 损失。"""
        loss_fn = MSELoss()
        pred = Tensor(NdArray([1.0, 2.0, 3.0]))
        target = Tensor(NdArray([2.0, 3.0, 4.0]))
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
        logits = Tensor(NdArray([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]]))
        # targets: [batch_size] (类别索引)
        targets = Tensor(NdArray([0.0, 1.0]))
        loss = loss_fn(logits, targets)
        assert loss.value.data[0] > 0

    def test_cross_entropy_backward_populates_logits_grad(self):
        """测试交叉熵损失会向 logits 回传正确梯度。"""
        logits = Tensor(NdArray([[2.0, 1.0, 0.1]]))
        targets = Tensor(NdArray([0.0]), requires_grad=False)

        loss = CrossEntropyLoss()(logits, targets)
        loss.backward()

        exp_values = [math.exp(0.0), math.exp(-1.0), math.exp(-1.9)]
        total = sum(exp_values)
        expected = [
            exp_values[0] / total - 1.0,
            exp_values[1] / total,
            exp_values[2] / total,
        ]

        assert logits.grad is not None
        for actual, target in zip(logits.grad.data, expected):
            assert abs(actual - target) < 1e-6

    def test_cross_entropy_reduction_none_keeps_batch_shape(self):
        """测试 reduction='none' 返回逐样本损失。"""
        logits = Tensor(NdArray([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]]))
        targets = Tensor(NdArray([0.0, 1.0]), requires_grad=False)

        loss = CrossEntropyLoss(reduction='none')(logits, targets)

        assert loss.value.shape.dims == (2,)


class TestBCELoss:
    """二元交叉熵损失的测试。"""

    def test_bce_forward(self):
        """测试 BCE 前向传播。"""
        probs = Tensor(NdArray([0.8, 0.3, 0.9]))
        targets = Tensor(NdArray([1.0, 0.0, 1.0]), requires_grad=False)

        loss = BCELoss()(probs, targets)

        assert loss.value.data[0] > 0

    def test_bce_backward_populates_input_grad(self):
        """测试 BCE 会向输入概率回传正确梯度。"""
        probs = Tensor(NdArray([0.8, 0.3, 0.9]))
        targets = Tensor(NdArray([1.0, 0.0, 1.0]), requires_grad=False)

        loss = BCELoss()(probs, targets)
        loss.backward()

        expected = [
            (0.8 - 1.0) / (0.8 * 0.2 * 3.0),
            (0.3 - 0.0) / (0.3 * 0.7 * 3.0),
            (0.9 - 1.0) / (0.9 * 0.1 * 3.0),
        ]

        assert probs.grad is not None
        for actual, target in zip(probs.grad.data, expected):
            assert abs(actual - target) < 1e-6


class TestLossProperties:
    """损失函数属性的测试。"""
    
    def test_loss_reduction_none(self):
        """测试不进行归约的损失。"""
        loss_fn = MSELoss(reduction='none')
        pred = Tensor(NdArray([1.0, 2.0, 3.0]))
        target = Tensor(NdArray([2.0, 3.0, 4.0]))
        loss = loss_fn(pred, target)
        # 应该返回每个样本的损失
        assert loss.value.shape.size == 3
        
    def test_loss_reduction_mean(self):
        """测试均值归约的损失。"""
        loss_fn = MSELoss(reduction='mean')
        pred = Tensor(NdArray([1.0, 2.0, 3.0]))
        target = Tensor(NdArray([2.0, 3.0, 4.0]))
        loss = loss_fn(pred, target)
        # 应该返回标量
        assert loss.value.shape.size == 1
        
    def test_loss_reduction_sum(self):
        """测试求和归约的损失。"""
        loss_fn = MSELoss(reduction='sum')
        pred = Tensor(NdArray([1.0, 2.0, 3.0]))
        target = Tensor(NdArray([2.0, 3.0, 4.0]))
        loss = loss_fn(pred, target)
        # 应该返回标量
        assert loss.value.shape.size == 1

    def test_bce_reduction_none_keeps_input_shape(self):
        """测试 BCE 在 reduction='none' 时保留输入形状。"""
        probs = Tensor(NdArray([[0.8], [0.3], [0.9]]))
        targets = Tensor(NdArray([[1.0], [0.0], [1.0]]), requires_grad=False)

        loss = BCELoss(reduction='none')(probs, targets)

        assert loss.value.shape.dims == (3, 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
