"""多维数组张量类。

本模块提供核心的 NdArray 类，表示多维数组并支持各种数学运算。

Author: TinyAI Team
Version: 0.1.0
"""

import math
from typing import Union, List, Tuple, Any, Callable
from tinytorch.ndarr.shape import Shape
from tinytorch.utils import random as tt_random

# Box-Muller 变换中 u1 的最小阈值，防止 math.log(0) 产生 -inf
_BOX_MULLER_MIN_U1 = 1e-10


class NdArray:
    """多维数组类，支持各种运算操作。
    
    NdArray 使用扁平列表存储数据（行优先布局），依赖 Shape 进行维度管理。
    所有操作都会创建新的 Tensor 对象。
    
    Attributes:
        data: 存储所有元素的扁平列表
        shape: 管理维度的 Shape 对象
        dtype: 数据类型（'float32' 或 'int32'）
    
    Example:
        >>> t = NdArray([[1, 2], [3, 4]])
        >>> print(t.shape)
        (2, 2)
        >>> t2 = t.add(NdArray([[1, 1], [1, 1]]))
        >>> print(t2.data)
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
        
        # 处理标量输入
        if isinstance(data, (int, float)):
            self.data = [self._cast(data)]
            self.shape = Shape((1,)) if shape is None else shape
            return
        
        # 处理嵌套列表输入
        if isinstance(data, list) and data and isinstance(data[0], list):
            self.shape, self.data = self._from_nested_list(data, dtype)
            return
        
        # 处理扁平列表输入
        if isinstance(data, list):
            if shape is None:
                self.shape = Shape((len(data),))
            else:
                # 如果 shape 是 tuple，转换为 Shape 对象
                if isinstance(shape, tuple):
                    shape = Shape(shape)
                self.shape = shape
                if len(data) != shape.size:
                    raise ValueError(f"Data size {len(data)} doesn't match shape size {shape.size}")
            
            self.data = [self._cast(x) for x in data]
            return
        
        raise ValueError(f"Unsupported data type: {type(data)}")

    def _cast(self, value) -> Union[float, int]:
        """将单个值转换为当前 dtype 对应的 Python 类型。"""
        return float(value) if self.dtype == 'float32' else int(value)

    @staticmethod
    def _cast_value(value, dtype: str) -> Union[float, int]:
        """将单个值转换为指定 dtype 对应的 Python 类型（静态版本）。"""
        return float(value) if dtype == 'float32' else int(value)

    @staticmethod
    def _ensure_shape(shape: Union[Tuple[int, ...], 'Shape']) -> 'Shape':
        """确保 shape 参数是 Shape 对象。"""
        if isinstance(shape, tuple):
            return Shape(shape)
        return shape

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
            """递归获取嵌套列表的形状，并校验各分支维度一致。"""
            if not isinstance(lst, list):
                return ()
            if not lst:
                return (0,)

            first = lst[0]
            if isinstance(first, list):
                child_shape = get_shape(first)
                for item in lst[1:]:
                    if not isinstance(item, list):
                        raise ValueError("Inconsistent nested list: mixed list/scalar elements")
                    if get_shape(item) != child_shape:
                        raise ValueError("Inconsistent nested list: ragged dimensions are not allowed")
                return (len(lst),) + child_shape

            for item in lst[1:]:
                if isinstance(item, list):
                    raise ValueError("Inconsistent nested list: mixed list/scalar elements")
            return (len(lst),)
        
        def flatten(lst):
            """递归展平嵌套列表。"""
            result = []
            for item in lst:
                if isinstance(item, list):
                    result.extend(flatten(item))
                else:
                    result.append(NdArray._cast_value(item, dtype))
            return result
        
        shape = Shape(get_shape(nested_list))
        flat_data = flatten(nested_list)
        return shape, flat_data
    
    # ==================== 工厂方法 ====================
    
    @staticmethod
    def zeros(shape: Union[Tuple[int, ...], Shape], dtype: str = 'float32') -> 'NdArray':
        """创建全零张量。
        
        Args:
            shape: 张量形状
            dtype: 数据类型
            
        Returns:
            全零张量
            
        Example:
            >>> t = NdArray.zeros((2, 3))
            >>> print(t.shape)
            (2, 3)
        """
        shape = NdArray._ensure_shape(shape)
        fill_value = 0.0 if dtype == 'float32' else 0
        data = [fill_value] * shape.size
        return NdArray(data, shape, dtype)
    
    @staticmethod
    def ones(shape: Union[Tuple[int, ...], Shape], dtype: str = 'float32') -> 'NdArray':
        """创建全一张量。
        
        Args:
            shape: 张量形状
            dtype: 数据类型
            
        Returns:
            全一张量
        """
        shape = NdArray._ensure_shape(shape)
        fill_value = 1.0 if dtype == 'float32' else 1
        data = [fill_value] * shape.size
        return NdArray(data, shape, dtype)
    
    @staticmethod
    def randn(shape: Union[Tuple[int, ...], Shape], seed: int = None, dtype: str = 'float32') -> 'NdArray':
        """创建服从标准正态分布的随机张量。
        
        Args:
            shape: 张量形状
            seed: 随机种子（可选）
            dtype: 数据类型
            
        Returns:
            随机张量
        """
        shape = NdArray._ensure_shape(shape)
        
        rng = tt_random.generator(seed) if seed is not None else tt_random
        
        # Box-Muller 变换生成正态分布
        data = []
        for _ in range(shape.size):
            u1 = rng.random()
            u2 = rng.random()
            # 将 u1 钳位到安全范围，防止 math.log(0) 产生 -inf
            u1 = max(u1, _BOX_MULLER_MIN_U1)
            z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            data.append(NdArray._cast_value(z0, dtype))
        
        return NdArray(data, shape, dtype)
    
    @staticmethod
    def uniform(low: float, high: float, shape: Union[Tuple[int, ...], Shape], 
                seed: int = None, dtype: str = 'float32') -> 'NdArray':
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
        shape = NdArray._ensure_shape(shape)
        
        rng = tt_random.generator(seed) if seed is not None else tt_random

        data = [NdArray._cast_value(rng.uniform(low, high), dtype) for _ in range(shape.size)]
        
        return NdArray(data, shape, dtype)
    
    # ==================== 基本运算 ====================
    
    def _elementwise_op(self, other: 'NdArray', op_name: str, op_func) -> 'NdArray':
        """逐元素二元运算的通用实现（处理广播）。
        
        Args:
            other: 右操作数
            op_name: 运算名称（用于错误信息）
            op_func: 二元运算函数 (x, y) -> result
        """
        if not isinstance(other, NdArray):
            raise TypeError(f"Unsupported type for {op_name}: {type(other)}")
        
        if self.shape == other.shape:
            result_data = [op_func(x, y) for x, y in zip(self.data, other.data)]
            return NdArray(result_data, self.shape, self.dtype)
        
        broadcast_shape = self.shape.broadcast_with(other.shape)
        t1_broadcast = self._broadcast_to(broadcast_shape)
        t2_broadcast = other._broadcast_to(broadcast_shape)
        
        result_data = [op_func(x, y) for x, y in zip(t1_broadcast.data, t2_broadcast.data)]
        return NdArray(result_data, broadcast_shape, self.dtype)
    
    def add(self, other: Union['NdArray', float, int]) -> 'NdArray':
        """逐元素加法。
        
        Args:
            other: 要相加的张量或标量
            
        Returns:
            运算结果张量
        """
        if isinstance(other, (int, float)):
            result_data = [x + other for x in self.data]
            return NdArray(result_data, self.shape, self.dtype)
        return self._elementwise_op(other, 'add', lambda x, y: x + y)
    
    def sub(self, other: Union['NdArray', float, int]) -> 'NdArray':
        """逐元素减法。"""
        if isinstance(other, (int, float)):
            result_data = [x - other for x in self.data]
            return NdArray(result_data, self.shape, self.dtype)
        return self._elementwise_op(other, 'sub', lambda x, y: x - y)
    
    def mul(self, other: Union['NdArray', float, int]) -> 'NdArray':
        """逐元素乘法。"""
        if isinstance(other, (int, float)):
            result_data = [x * other for x in self.data]
            return NdArray(result_data, self.shape, self.dtype)
        return self._elementwise_op(other, 'mul', lambda x, y: x * y)
    
    def div(self, other: Union['NdArray', float, int]) -> 'NdArray':
        """逐元素除法。

        除零行为与主流框架一致：返回 inf/-inf/nan 而非抛出异常。
        """
        if isinstance(other, (int, float)):
            if other == 0:
                result_data = [math.copysign(math.inf, x) if x != 0 else math.nan for x in self.data]
            else:
                result_data = [x / other for x in self.data]
            return NdArray(result_data, self.shape, self.dtype)
        
        def safe_div(x, y):
            if y == 0:
                return math.copysign(math.inf, x) if x != 0 else math.nan
            return x / y
        
        return self._elementwise_op(other, 'div', safe_div)
    
    def neg(self) -> 'NdArray':
        """取负。"""
        result_data = [-x for x in self.data]
        return NdArray(result_data, self.shape, self.dtype)
    
    # ==================== 矩阵运算 ====================
    
    def matmul(self, other: 'NdArray') -> 'NdArray':
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
            
            zero_val = 0.0 if self.dtype == 'float32' else 0
            result_data = [zero_val] * (m * n)
            for i in range(m):
                for j in range(n):
                    sum_val = zero_val
                    for k in range(k1):
                        sum_val += self.data[i * k1 + k] * other.data[k * n + j]
                    result_data[i * n + j] = sum_val
            
            return NdArray(result_data, Shape((m, n)), self.dtype)
        
        raise NotImplementedError("Only 2D matmul is currently supported")
    
    def transpose(self, axes: Tuple[int, ...] = None) -> 'NdArray':
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
            return NdArray(result_data, new_shape, self.dtype)
        
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
        
        return NdArray(result_data, new_shape, self.dtype)
    
    @staticmethod
    def _linear_to_indices(linear_idx: int, shape: Shape) -> Tuple[int, ...]:
        """将线性索引转换为多维索引。"""
        indices = []
        for stride in shape.strides:
            idx = linear_idx // stride
            indices.append(idx)
            linear_idx -= idx * stride
        return tuple(indices)
    
    def reshape(self, new_shape: Union[Tuple[int, ...], List[int]]) -> 'NdArray':
        """将张量重塑为新维度。
        
        Args:
            new_shape: 新形状（可包含 -1 进行推断）
            
        Returns:
            重塑后的张量（共享相同数据）
        """
        if isinstance(new_shape, list):
            new_shape = tuple(new_shape)
        
        reshaped_shape = self.shape.reshape(new_shape)
        return NdArray(self.data.copy(), reshaped_shape, self.dtype)
    
    # ==================== 归约运算 ====================
    
    def sum(self, axis: int = None, keepdims: bool = False) -> 'NdArray':
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
            return NdArray([result], new_shape, self.dtype)
        
        # 沿特定轴求和
        if axis < 0:
            axis = self.shape.ndim + axis
        if axis < 0 or axis >= self.shape.ndim:
            raise ValueError(f"axis {axis} out of range for {self.shape.ndim}D ndarr")
        
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
        
        return NdArray(result_data, new_shape, self.dtype)
    
    def mean(self, axis: int = None, keepdims: bool = False) -> 'NdArray':
        """求张量元素的平均值。"""
        sum_tensor = self.sum(axis, keepdims)
        if axis is None:
            count = self.shape.size
        else:
            count = self.shape.dims[axis if axis >= 0 else self.shape.ndim + axis]
        
        result_data = [x / count for x in sum_tensor.data]
        return NdArray(result_data, sum_tensor.shape, self.dtype)
    
    def max(self, axis: int = None, keepdims: bool = False) -> 'NdArray':
        """求张量元素的最大值。
        
        Args:
            axis: 沿哪个轴求最大值（None 表示所有）
            keepdims: 是否保持被缩减的维度
            
        Returns:
            最大值结果
        """
        if axis is None:
            result = max(self.data)
            if keepdims:
                new_shape = Shape((1,) * self.shape.ndim)
            else:
                new_shape = Shape((1,))
            return NdArray([result], new_shape, self.dtype)
        
        if axis < 0:
            axis = self.shape.ndim + axis
        if axis < 0 or axis >= self.shape.ndim:
            raise ValueError(f"axis {axis} out of range for {self.shape.ndim}D ndarr")
        
        new_dims = list(self.shape.dims)
        if keepdims:
            new_dims[axis] = 1
        else:
            new_dims.pop(axis)
        if not new_dims:
            new_dims = [1]
        new_shape = Shape(tuple(new_dims))
        
        result_data = [float('-inf')] * new_shape.size
        for i in range(self.shape.size):
            old_indices = self._linear_to_indices(i, self.shape)
            new_indices = list(old_indices)
            if keepdims:
                new_indices[axis] = 0
            else:
                new_indices.pop(axis)
            new_idx = new_shape.linear_index(tuple(new_indices))
            if self.data[i] > result_data[new_idx]:
                result_data[new_idx] = self.data[i]
        
        return NdArray(result_data, new_shape, self.dtype)
    
    def min(self, axis: int = None, keepdims: bool = False) -> 'NdArray':
        """求张量元素的最小值。
        
        Args:
            axis: 沿哪个轴求最小值（None 表示所有）
            keepdims: 是否保持被缩减的维度
            
        Returns:
            最小值结果
        """
        if axis is None:
            result = min(self.data)
            if keepdims:
                new_shape = Shape((1,) * self.shape.ndim)
            else:
                new_shape = Shape((1,))
            return NdArray([result], new_shape, self.dtype)
        
        if axis < 0:
            axis = self.shape.ndim + axis
        if axis < 0 or axis >= self.shape.ndim:
            raise ValueError(f"axis {axis} out of range for {self.shape.ndim}D ndarr")
        
        new_dims = list(self.shape.dims)
        if keepdims:
            new_dims[axis] = 1
        else:
            new_dims.pop(axis)
        if not new_dims:
            new_dims = [1]
        new_shape = Shape(tuple(new_dims))
        
        result_data = [float('inf')] * new_shape.size
        for i in range(self.shape.size):
            old_indices = self._linear_to_indices(i, self.shape)
            new_indices = list(old_indices)
            if keepdims:
                new_indices[axis] = 0
            else:
                new_indices.pop(axis)
            new_idx = new_shape.linear_index(tuple(new_indices))
            if self.data[i] < result_data[new_idx]:
                result_data[new_idx] = self.data[i]
        
        return NdArray(result_data, new_shape, self.dtype)
    
    # ==================== 数学函数 ====================
    
    def exp(self) -> 'NdArray':
        """逐元素指数运算。"""
        result_data = []
        for x in self.data:
            # 避免 math.exp 在极值上直接抛出 OverflowError。
            if x > 709.0:
                result_data.append(float('inf'))
            elif x < -745.0:
                result_data.append(0.0)
            else:
                result_data.append(math.exp(x))
        return NdArray(result_data, self.shape, self.dtype)
    
    def log(self) -> 'NdArray':
        """逐元素自然对数。"""
        result_data = []
        for x in self.data:
            if x <= 0:
                raise ValueError("log requires positive values")
            result_data.append(math.log(x))
        return NdArray(result_data, self.shape, self.dtype)
    
    def sqrt(self) -> 'NdArray':
        """逐元素平方根。"""
        result_data = []
        for x in self.data:
            if x < 0:
                raise ValueError("sqrt requires non-negative values")
            result_data.append(math.sqrt(x))
        return NdArray(result_data, self.shape, self.dtype)
    
    def pow(self, exponent: float) -> 'NdArray':
        """逐元素幂运算。"""
        result_data = [x ** exponent for x in self.data]
        return NdArray(result_data, self.shape, self.dtype)
    
    # ==================== 激活函数 ====================
    
    def relu(self) -> 'NdArray':
        """ReLU 激活函数。"""
        result_data = [max(0.0, x) for x in self.data]
        return NdArray(result_data, self.shape, self.dtype)
    
    def sigmoid(self) -> 'NdArray':
        """Sigmoid 激活函数。"""
        # 使用分段形式提升数值稳定性，避免 exp(-x) / exp(x) 溢出。
        result_data = []
        for x in self.data:
            if x >= 0:
                z = math.exp(-x)
                result_data.append(1.0 / (1.0 + z))
            else:
                z = math.exp(x)
                result_data.append(z / (1.0 + z))
        return NdArray(result_data, self.shape, self.dtype)
    
    def tanh(self) -> 'NdArray':
        """Tanh 激活函数。"""
        result_data = [math.tanh(x) for x in self.data]
        return NdArray(result_data, self.shape, self.dtype)
    
    # ==================== 广播辅助方法 ====================
    
    def _broadcast_to(self, target_shape: Shape) -> 'NdArray':
        """将当前张量广播到目标形状。
        
        Args:
            target_shape: 要广播到的目标形状
            
        Returns:
            广播后的张量
        """
        if self.shape == target_shape:
            return NdArray(self.data.copy(), self.shape, self.dtype)
        
        # 在左侧填充维度
        ndim_diff = target_shape.ndim - self.shape.ndim
        source_dims = (1,) * ndim_diff + self.shape.dims
        
        # 预计算源 strides，避免在循环内反复创建 Shape 对象
        src_strides = self.shape.strides
        
        result_data = [0.0] * target_shape.size
        
        for i in range(target_shape.size):
            target_indices = self._linear_to_indices(i, target_shape)
            
            # 直接映射到原始数据索引，跳过中间 Shape 创建
            src_linear_idx = 0
            for j in range(self.shape.ndim):
                src_dim = source_dims[ndim_diff + j]
                tgt_idx = target_indices[ndim_diff + j]
                if src_dim != 1:
                    src_linear_idx += tgt_idx * src_strides[j]
            
            result_data[i] = self.data[src_linear_idx]
        
        return NdArray(result_data, target_shape, self.dtype)
    
    # ==================== 工具方法 ====================

    def to_list(self) -> List:
        """将张量转换为嵌套列表。"""
        if not self.shape.dims:
            return self.data[0]

        def build(offset: int, dims: Tuple[int, ...]):
            if len(dims) == 1:
                end = offset + dims[0]
                return self.data[offset:end]

            step = 1
            for dim in dims[1:]:
                step *= dim

            return [
                build(offset + i * step, dims[1:])
                for i in range(dims[0])
            ]

        return build(0, self.shape.dims)
    
    def copy(self) -> 'NdArray':
        """创建当前张量的深拷贝。"""
        return NdArray(self.data.copy(), self.shape, self.dtype)
    
    def __repr__(self) -> str:
        """字符串表示。"""
        return f"NdArray(shape={self.shape}, dtype={self.dtype}, data={self.data[:10]}...)" if len(self.data) > 10 else f"NdArray(shape={self.shape}, dtype={self.dtype}, data={self.data})"
    
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
