"""多维数组张量类。

本模块提供核心的 Tensor 类，表示多维数组并支持各种数学运算。

Author: TinyAI Team
Version: 0.1.0
"""

import math
import random
from typing import Union, List, Tuple, Any
from tinytorch.tensor.shape import Shape


class Tensor:
    """多维数组类，支持各种运算操作。
    
    Tensor 使用扁平列表存储数据（行优先布局），依赖 Shape 进行维度管理。
    所有操作都会创建新的 Tensor 对象。
    
    Attributes:
        data: 存储所有元素的扁平列表
        shape: 管理维度的 Shape 对象
        dtype: 数据类型（'float32' 或 'int32'）
    
    Example:
        >>> t = Tensor([[1, 2], [3, 4]])
        >>> print(t.shape)
        (2, 2)
        >>> t2 = t.add(Tensor([[1, 1], [1, 1]]))
        >>> print(t2.data)
        [2.0, 3.0, 4.0, 5.0]
    """
    
    def __init__(self, data: Union[List, float, int], shape: Shape = None, dtype: str = 'float32'):
        """初始化张量。
        
        Args:
            data: 输入数据（嵌套列表、扁平列表或标量）
            shape: Shape 对象（如果为 None 则自动推断）
            dtype: 数据类型（'float32' 或 'int32'）
        """
        if dtype not in ['float32', 'int32']:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        self.dtype = dtype
        
        # Handle scalar input
        if isinstance(data, (int, float)):
            self.data = [float(data) if dtype == 'float32' else int(data)]
            self.shape = Shape((1,)) if shape is None else shape
            return
        
        # Handle nested list input
        if isinstance(data, list) and data and isinstance(data[0], list):
            self.shape, self.data = self._from_nested_list(data, dtype)
            return
        
        # Handle flat list input
        if isinstance(data, list):
            if shape is None:
                self.shape = Shape((len(data),))
            else:
                self.shape = shape
                if len(data) != shape.size:
                    raise ValueError(f"Data size {len(data)} doesn't match shape size {shape.size}")
            
            self.data = [float(x) if dtype == 'float32' else int(x) for x in data]
            return
        
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    @staticmethod
    def _from_nested_list(nested_list: List, dtype: str) -> Tuple[Shape, List]:
        """将嵌套列表转换为扁平列表并推断形状。
        
        Args:
            nested_list: 嵌套列表结构
            dtype: 数据类型
            
        Returns:
            (Shape, 扁平列表) 元组
        """
        def get_shape(lst):
            """递归获取嵌套列表的形状。"""
            if not isinstance(lst, list):
                return ()
            if not lst:
                return (0,)
            return (len(lst),) + get_shape(lst[0])
        
        def flatten(lst):
            """递归展平嵌套列表。"""
            result = []
            for item in lst:
                if isinstance(item, list):
                    result.extend(flatten(item))
                else:
                    result.append(float(item) if dtype == 'float32' else int(item))
            return result
        
        shape = Shape(get_shape(nested_list))
        flat_data = flatten(nested_list)
        return shape, flat_data
    
    # ==================== 工厂方法 ====================
    
    @staticmethod
    def zeros(shape: Union[Tuple[int, ...], Shape], dtype: str = 'float32') -> 'Tensor':
        """创建全零张量。
        
        Args:
            shape: 张量形状
            dtype: 数据类型
            
        Returns:
            全零张量
            
        Example:
            >>> t = Tensor.zeros((2, 3))
            >>> print(t.shape)
            (2, 3)
        """
        if isinstance(shape, tuple):
            shape = Shape(shape)
        data = [0.0 if dtype == 'float32' else 0] * shape.size
        return Tensor(data, shape, dtype)
    
    @staticmethod
    def ones(shape: Union[Tuple[int, ...], Shape], dtype: str = 'float32') -> 'Tensor':
        """创建全一张量。
        
        Args:
            shape: 张量形状
            dtype: 数据类型
            
        Returns:
            全一张量
        """
        if isinstance(shape, tuple):
            shape = Shape(shape)
        data = [1.0 if dtype == 'float32' else 1] * shape.size
        return Tensor(data, shape, dtype)
    
    @staticmethod
    def randn(shape: Union[Tuple[int, ...], Shape], seed: int = None, dtype: str = 'float32') -> 'Tensor':
        """创建服从标准正态分布的随机张量。
        
        Args:
            shape: 张量形状
            seed: 随机种子（可选）
            dtype: 数据类型
            
        Returns:
            随机张量
        """
        if isinstance(shape, tuple):
            shape = Shape(shape)
        
        if seed is not None:
            random.seed(seed)
        
        # Box-Muller 变换生成正态分布
        data = []
        for _ in range(shape.size):
            u1 = random.random()
            u2 = random.random()
            z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            data.append(float(z0) if dtype == 'float32' else int(z0))
        
        return Tensor(data, shape, dtype)
    
    @staticmethod
    def uniform(low: float, high: float, shape: Union[Tuple[int, ...], Shape], 
                seed: int = None, dtype: str = 'float32') -> 'Tensor':
        """创建服从均匀分布的随机张量。
        
        Args:
            low: 下界
            high: 上界
            shape: 张量形状
            seed: 随机种子（可选）
            dtype: 数据类型
            
        Returns:
            均匀分布随机张量
        """
        if isinstance(shape, tuple):
            shape = Shape(shape)
        
        if seed is not None:
            random.seed(seed)
        
        data = [random.uniform(low, high) for _ in range(shape.size)]
        if dtype == 'int32':
            data = [int(x) for x in data]
        
        return Tensor(data, shape, dtype)
    
    # ==================== 基本运算 ====================
    
    def add(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """逐元素加法。
        
        Args:
            other: 要相加的张量或标量
            
        Returns:
            运算结果张量
        """
        if isinstance(other, (int, float)):
            result_data = [x + other for x in self.data]
            return Tensor(result_data, self.shape, self.dtype)
        
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported type for add: {type(other)}")
        
        # 处理广播
        if self.shape == other.shape:
            result_data = [x + y for x, y in zip(self.data, other.data)]
            return Tensor(result_data, self.shape, self.dtype)
        
        # 广播情况
        broadcast_shape = self.shape.broadcast_with(other.shape)
        t1_broadcast = self._broadcast_to(broadcast_shape)
        t2_broadcast = other._broadcast_to(broadcast_shape)
        
        result_data = [x + y for x, y in zip(t1_broadcast.data, t2_broadcast.data)]
        return Tensor(result_data, broadcast_shape, self.dtype)
    
    def sub(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """逐元素减法。"""
        if isinstance(other, (int, float)):
            result_data = [x - other for x in self.data]
            return Tensor(result_data, self.shape, self.dtype)
        
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported type for sub: {type(other)}")
        
        if self.shape == other.shape:
            result_data = [x - y for x, y in zip(self.data, other.data)]
            return Tensor(result_data, self.shape, self.dtype)
        
        broadcast_shape = self.shape.broadcast_with(other.shape)
        t1_broadcast = self._broadcast_to(broadcast_shape)
        t2_broadcast = other._broadcast_to(broadcast_shape)
        
        result_data = [x - y for x, y in zip(t1_broadcast.data, t2_broadcast.data)]
        return Tensor(result_data, broadcast_shape, self.dtype)
    
    def mul(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """逐元素乘法。"""
        if isinstance(other, (int, float)):
            result_data = [x * other for x in self.data]
            return Tensor(result_data, self.shape, self.dtype)
        
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported type for mul: {type(other)}")
        
        if self.shape == other.shape:
            result_data = [x * y for x, y in zip(self.data, other.data)]
            return Tensor(result_data, self.shape, self.dtype)
        
        broadcast_shape = self.shape.broadcast_with(other.shape)
        t1_broadcast = self._broadcast_to(broadcast_shape)
        t2_broadcast = other._broadcast_to(broadcast_shape)
        
        result_data = [x * y for x, y in zip(t1_broadcast.data, t2_broadcast.data)]
        return Tensor(result_data, broadcast_shape, self.dtype)
    
    def div(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """逐元素除法。"""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ValueError("Division by zero")
            result_data = [x / other for x in self.data]
            return Tensor(result_data, self.shape, self.dtype)
        
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported type for div: {type(other)}")
        
        if self.shape == other.shape:
            result_data = []
            for x, y in zip(self.data, other.data):
                if y == 0:
                    raise ValueError("Division by zero")
                result_data.append(x / y)
            return Tensor(result_data, self.shape, self.dtype)
        
        broadcast_shape = self.shape.broadcast_with(other.shape)
        t1_broadcast = self._broadcast_to(broadcast_shape)
        t2_broadcast = other._broadcast_to(broadcast_shape)
        
        result_data = []
        for x, y in zip(t1_broadcast.data, t2_broadcast.data):
            if y == 0:
                raise ValueError("Division by zero")
            result_data.append(x / y)
        return Tensor(result_data, broadcast_shape, self.dtype)
    
    def neg(self) -> 'Tensor':
        """取负。"""
        result_data = [-x for x in self.data]
        return Tensor(result_data, self.shape, self.dtype)
    
    # ==================== 矩阵运算 ====================
    
    def matmul(self, other: 'Tensor') -> 'Tensor':
        """矩阵乘法。
        
        Args:
            other: 右操作数张量
            
        Returns:
            矩阵乘法结果
            
        Raises:
            ValueError: 当形状不匹配时
        """
        if self.shape.ndim < 2 or other.shape.ndim < 2:
            raise ValueError("matmul requires at least 2D tensors")
        
        # 为简单起见，处理 2D 情况
        if self.shape.ndim == 2 and other.shape.ndim == 2:
            m, k1 = self.shape.dims
            k2, n = other.shape.dims
            
            if k1 != k2:
                raise ValueError(f"Incompatible shapes for matmul: {self.shape.dims} and {other.shape.dims}")
            
            result_data = [0.0] * (m * n)
            for i in range(m):
                for j in range(n):
                    sum_val = 0.0
                    for k in range(k1):
                        sum_val += self.data[i * k1 + k] * other.data[k * n + j]
                    result_data[i * n + j] = sum_val
            
            return Tensor(result_data, Shape((m, n)), self.dtype)
        
        raise NotImplementedError("Only 2D matmul is currently supported")
    
    def transpose(self, axes: Tuple[int, ...] = None) -> 'Tensor':
        """转置张量维度。
        
        Args:
            axes: 维度的排列（None 表示默认反转）
            
        Returns:
            转置后的张量
        """
        new_shape = self.shape.transpose(axes)
        
        # 对于 2D 情况（最常见）
        if self.shape.ndim == 2 and axes is None:
            m, n = self.shape.dims
            result_data = [0.0] * (m * n)
            for i in range(m):
                for j in range(n):
                    result_data[j * m + i] = self.data[i * n + j]
            return Tensor(result_data, new_shape, self.dtype)
        
        # 通用情况：使用索引映射
        result_data = [0.0] * self.shape.size
        for i in range(self.shape.size):
            # 将线性索引转换为多维索引
            old_indices = self._linear_to_indices(i, self.shape)
            
            # 应用排列
            if axes is None:
                new_indices = tuple(reversed(old_indices))
            else:
                new_indices = tuple(old_indices[axes[j]] for j in range(len(axes)))
            
            # 转换回新形状中的线性索引
            new_idx = new_shape.linear_index(new_indices)
            result_data[new_idx] = self.data[i]
        
        return Tensor(result_data, new_shape, self.dtype)
    
    @staticmethod
    def _linear_to_indices(linear_idx: int, shape: Shape) -> Tuple[int, ...]:
        """将线性索引转换为多维索引。"""
        indices = []
        for stride in shape.strides:
            idx = linear_idx // stride
            indices.append(idx)
            linear_idx -= idx * stride
        return tuple(indices)
    
    def reshape(self, new_shape: Union[Tuple[int, ...], List[int]]) -> 'Tensor':
        """将张量重塑为新维度。
        
        Args:
            new_shape: 新形状（可包含 -1 进行推断）
            
        Returns:
            重塑后的张量（共享相同数据）
        """
        if isinstance(new_shape, list):
            new_shape = tuple(new_shape)
        
        reshaped_shape = self.shape.reshape(new_shape)
        return Tensor(self.data.copy(), reshaped_shape, self.dtype)
    
    # ==================== 归约运算 ====================
    
    def sum(self, axis: int = None, keepdims: bool = False) -> 'Tensor':
        """求和张量元素。
        
        Args:
            axis: 沿哪个轴求和（None 表示所有）
            keepdims: 是否保持被缩减的维度
            
        Returns:
            求和结果
        """
        if axis is None:
            # 求所有元素的和
            result = sum(self.data)
            if keepdims:
                new_shape = Shape((1,) * self.shape.ndim)
            else:
                new_shape = Shape((1,))
            return Tensor([result], new_shape, self.dtype)
        
        # 沿特定轴求和
        if axis < 0:
            axis = self.shape.ndim + axis
        if axis < 0 or axis >= self.shape.ndim:
            raise ValueError(f"axis {axis} out of range for {self.shape.ndim}D tensor")
        
        # 计算新形状
        new_dims = list(self.shape.dims)
        if keepdims:
            new_dims[axis] = 1
        else:
            new_dims.pop(axis)
        
        if not new_dims:
            new_dims = [1]
        new_shape = Shape(tuple(new_dims))
        
        # 执行缩减
        result_size = new_shape.size
        result_data = [0.0] * result_size
        
        for i in range(self.shape.size):
            old_indices = self._linear_to_indices(i, self.shape)
            new_indices = list(old_indices)
            if keepdims:
                new_indices[axis] = 0
            else:
                new_indices.pop(axis)
            new_idx = new_shape.linear_index(tuple(new_indices))
            result_data[new_idx] += self.data[i]
        
        return Tensor(result_data, new_shape, self.dtype)
    
    def mean(self, axis: int = None, keepdims: bool = False) -> 'Tensor':
        """求张量元素的平均值。"""
        sum_tensor = self.sum(axis, keepdims)
        if axis is None:
            count = self.shape.size
        else:
            count = self.shape.dims[axis if axis >= 0 else self.shape.ndim + axis]
        
        result_data = [x / count for x in sum_tensor.data]
        return Tensor(result_data, sum_tensor.shape, self.dtype)
    
    # ==================== 数学函数 ====================
    
    def exp(self) -> 'Tensor':
        """逐元素指数运算。"""
        result_data = [math.exp(x) for x in self.data]
        return Tensor(result_data, self.shape, self.dtype)
    
    def log(self) -> 'Tensor':
        """逐元素自然对数。"""
        result_data = []
        for x in self.data:
            if x <= 0:
                raise ValueError("log requires positive values")
            result_data.append(math.log(x))
        return Tensor(result_data, self.shape, self.dtype)
    
    def sqrt(self) -> 'Tensor':
        """逐元素平方根。"""
        result_data = []
        for x in self.data:
            if x < 0:
                raise ValueError("sqrt requires non-negative values")
            result_data.append(math.sqrt(x))
        return Tensor(result_data, self.shape, self.dtype)
    
    def pow(self, exponent: float) -> 'Tensor':
        """逐元素幂运算。"""
        result_data = [x ** exponent for x in self.data]
        return Tensor(result_data, self.shape, self.dtype)
    
    # ==================== 激活函数 ====================
    
    def relu(self) -> 'Tensor':
        """ReLU 激活函数。"""
        result_data = [max(0.0, x) for x in self.data]
        return Tensor(result_data, self.shape, self.dtype)
    
    def sigmoid(self) -> 'Tensor':
        """Sigmoid 激活函数。"""
        result_data = [1.0 / (1.0 + math.exp(-x)) for x in self.data]
        return Tensor(result_data, self.shape, self.dtype)
    
    def tanh(self) -> 'Tensor':
        """Tanh 激活函数。"""
        result_data = [math.tanh(x) for x in self.data]
        return Tensor(result_data, self.shape, self.dtype)
    
    # ==================== 广播辅助方法 ====================
    
    def _broadcast_to(self, target_shape: Shape) -> 'Tensor':
        """将当前张量广播到目标形状。
        
        Args:
            target_shape: 要广播到的目标形状
            
        Returns:
            广播后的张量
        """
        if self.shape == target_shape:
            return Tensor(self.data.copy(), self.shape, self.dtype)
        
        # 在左侧填充维度
        ndim_diff = target_shape.ndim - self.shape.ndim
        source_dims = (1,) * ndim_diff + self.shape.dims
        
        result_data = [0.0] * target_shape.size
        
        for i in range(target_shape.size):
            target_indices = self._linear_to_indices(i, target_shape)
            
            # 映射到源索引（广播规则）
            source_indices = []
            for j, (src_dim, tgt_idx) in enumerate(zip(source_dims, target_indices)):
                if src_dim == 1:
                    source_indices.append(0)
                else:
                    source_indices.append(tgt_idx)
            
            # 获取源线性索引
            temp_shape = Shape(source_dims)
            src_linear_idx = temp_shape.linear_index(tuple(source_indices))
            
            # 映射回原始数据
            if ndim_diff > 0:
                # 需要调整填充
                orig_indices = source_indices[ndim_diff:]
                src_linear_idx = self.shape.linear_index(tuple(orig_indices))
            
            result_data[i] = self.data[src_linear_idx]
        
        return Tensor(result_data, target_shape, self.dtype)
    
    # ==================== 工具方法 ====================
    
    def copy(self) -> 'Tensor':
        """创建当前张量的深拷贝。"""
        return Tensor(self.data.copy(), self.shape, self.dtype)
    
    def __repr__(self) -> str:
        """字符串表示。"""
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, data={self.data[:10]}...)" if len(self.data) > 10 else f"Tensor(shape={self.shape}, dtype={self.dtype}, data={self.data})"
    
    def __str__(self) -> str:
        """字符串表示。"""
        return self.__repr__()
    
    # ==================== 运算符重载 ====================
    
    def __add__(self, other):
        """加法运算符。"""
        return self.add(other)
    
    def __sub__(self, other):
        """减法运算符。"""
        return self.sub(other)
    
    def __mul__(self, other):
        """乘法运算符。"""
        return self.mul(other)
    
    def __truediv__(self, other):
        """除法运算符。"""
        return self.div(other)
    
    def __neg__(self):
        """取负运算符。"""
        return self.neg()
    
    def __matmul__(self, other):
        """矩阵乘法运算符。"""
        return self.matmul(other)
