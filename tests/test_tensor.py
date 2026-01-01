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


class TestTensorOperations:
    """张量运算的测试。"""
    
    def test_tensor_sub(self):
        """测试张量减法。"""
        t1 = Tensor([5, 6, 7])
        t2 = Tensor([1, 2, 3])
        result = t1.sub(t2)
        assert result.data == [4.0, 4.0, 4.0]
        
    def test_tensor_div(self):
        """测试张量除法。"""
        t1 = Tensor([10, 20, 30])
        t2 = Tensor([2, 4, 5])
        result = t1.div(t2)
        assert result.data == [5.0, 5.0, 6.0]
        
    def test_tensor_pow(self):
        """测试张量幂运算。"""
        t = Tensor([2, 3, 4])
        result = t.pow(2)
        assert result.data == [4.0, 9.0, 16.0]
        
    def test_tensor_sqrt(self):
        """测试张量平方根。"""
        t = Tensor([4, 9, 16])
        result = t.sqrt()
        assert result.data == [2.0, 3.0, 4.0]
        
    def test_tensor_exp(self):
        """测试张量指数。"""
        t = Tensor([0, 1, 2])
        result = t.exp()
        # e^0 = 1, e^1 ≈ 2.718, e^2 ≈ 7.389
        assert abs(result.data[0] - 1.0) < 0.01
        
    def test_tensor_log(self):
        """测试张量对数。"""
        t = Tensor([1, 2.718, 7.389])
        result = t.log()
        # ln(1) = 0
        assert abs(result.data[0] - 0.0) < 0.01
        
    def test_tensor_sum(self):
        """测试张量求和。"""
        t = Tensor([1, 2, 3, 4])
        result = t.sum()
        assert result.data[0] == 10.0
        
    def test_tensor_mean(self):
        """测试张量均值。"""
        t = Tensor([2, 4, 6, 8])
        result = t.mean()
        assert result.data[0] == 5.0
        
    def test_tensor_max(self):
        """测试张量最大值。"""
        t = Tensor([1, 5, 3, 9, 2])
        result = t.max()
        assert result.data[0] == 9.0
        
    def test_tensor_min(self):
        """测试张量最小值。"""
        t = Tensor([5, 2, 8, 1, 6])
        result = t.min()
        assert result.data[0] == 1.0
        
    def test_tensor_relu(self):
        """测试张量 ReLU。"""
        t = Tensor([-2, -1, 0, 1, 2])
        result = t.relu()
        assert result.data == [0.0, 0.0, 0.0, 1.0, 2.0]


class TestTensorProperties:
    """张量属性的测试。"""
    
    def test_tensor_dtype(self):
        """测试张量数据类型。"""
        t = Tensor([1.0, 2.0, 3.0], dtype='float32')
        assert t.dtype == 'float32'
        
    def test_tensor_ndim(self):
        """测试张量维数。"""
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([[1, 2], [3, 4]])
        t3 = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert t1.shape.ndim == 1
        assert t2.shape.ndim == 2
        assert t3.shape.ndim == 3
        
    def test_tensor_size(self):
        """测试张量大小。"""
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([[1, 2], [3, 4]])
        assert t1.shape.size == 3
        assert t2.shape.size == 4
        
    def test_tensor_copy(self):
        """测试张量复制。"""
        t1 = Tensor([1, 2, 3])
        t2 = t1.copy()
        assert t2.data == t1.data
        assert t2 is not t1  # 不同对象


class TestTensorBroadcast:
    """张量广播的测试。"""
    
    def test_broadcast_add(self):
        """测试广播加法。"""
        t1 = Tensor([[1, 2], [3, 4]])
        t2 = Tensor([10, 20])
        result = t1.add(t2)
        # 应该将 t2 广播到每一行
        assert result.shape.dims == (2, 2)


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
