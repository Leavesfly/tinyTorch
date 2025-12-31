"""测试 Tensor 基本功能。

Author: TinyAI Team
"""

import pytest
from tinytorch.tensor import Tensor, Shape


class TestTensor:
    """Tensor 类的测试。"""
    
    def test_tensor_creation(self):
        """测试张量创建。"""
        # 从列表创建
        t = Tensor([1, 2, 3, 4])
        assert t.shape.dims == (4,)
        assert len(t.data) == 4
        
    def test_tensor_shape(self):
        """测试张量形状。"""
        t = Tensor([[1, 2], [3, 4]])
        assert t.shape.dims == (2, 2)
        assert t.shape.size == 4
        
    def test_tensor_zeros(self):
        """测试全零张量。"""
        t = Tensor.zeros((2, 3))
        assert t.shape.dims == (2, 3)
        assert all(x == 0.0 for x in t.data)
        
    def test_tensor_ones(self):
        """测试全一张量。"""
        t = Tensor.ones((3, 2))
        assert t.shape.dims == (3, 2)
        assert all(x == 1.0 for x in t.data)
        
    def test_tensor_add(self):
        """测试张量加法。"""
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        result = t1.add(t2)
        assert result.data == [5.0, 7.0, 9.0]
        
    def test_tensor_mul(self):
        """测试张量乘法。"""
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([2, 3, 4])
        result = t1.mul(t2)
        assert result.data == [2.0, 6.0, 12.0]
        
    def test_tensor_reshape(self):
        """测试张量重塑。"""
        t = Tensor([1, 2, 3, 4, 5, 6])
        reshaped = t.reshape((2, 3))
        assert reshaped.shape.dims == (2, 3)
        
    def test_tensor_transpose(self):
        """测试张量转置。"""
        t = Tensor([[1, 2], [3, 4]])
        transposed = t.transpose()
        assert transposed.shape.dims == (2, 2)


class TestShape:
    """Shape 类的测试。"""
    
    def test_shape_creation(self):
        """测试形状创建。"""
        s = Shape((2, 3, 4))
        assert s.dims == (2, 3, 4)
        assert s.ndim == 3
        assert s.size == 24
        
    def test_shape_broadcast(self):
        """测试形状广播。"""
        s1 = Shape((2, 3))
        s2 = Shape((3,))
        result = s1.broadcast_with(s2)
        assert result.dims == (2, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
