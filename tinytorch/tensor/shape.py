"""Shape 类用于管理张量的维度和内存布局。

本模块提供 Shape 类，处理多维数组的形状信息、步长计算和广播逻辑。

Author: TinyAI Team
Version: 0.1.0
"""

from typing import Tuple, List, Union


class Shape:
    """张量的形状管理类。
    
    Shape 类封装维度信息，并提供形状操作、广播和索引计算的实用工具。
    
    Attributes:
        dims: 维度大小的元组
        ndim: 维度数量
        size: 元素总数
        strides: 每个维度的步长元组
    
    Example:
        >>> shape = Shape((2, 3, 4))
        >>> print(shape.dims)
        (2, 3, 4)
        >>> print(shape.size)
        24
        >>> print(shape.strides)
        (12, 4, 1)
    """
    
    def __init__(self, dims: Union[Tuple[int, ...], List[int]]):
        """初始化 Shape 对象。
        
        Args:
            dims: 维度大小，可以是元组或列表
            
        Raises:
            ValueError: 如果 dims 为空或包含非正值
        """
        if not dims:
            raise ValueError("Shape cannot be empty")
        
        # 转换为元组并验证
        if isinstance(dims, list):
            dims = tuple(dims)
        
        if not all(isinstance(d, int) and d > 0 for d in dims):
            raise ValueError("All dimensions must be positive integers")
        
        self._dims = dims
        self._ndim = len(dims)
        self._size = 1
        for d in dims:
            self._size *= d
        self._strides = self._compute_strides(dims)
    
    @staticmethod
    def _compute_strides(dims: Tuple[int, ...]) -> Tuple[int, ...]:
        """计算行优先（C 风格）内存布局的步长。
        
        对于形状 (d0, d1, d2, ..., dn)，步长计算为：
        stride[i] = product of dims[i+1:]
        
        Args:
            dims: 维度大小
            
        Returns:
            步长元组
            
        Example:
            >>> Shape._compute_strides((2, 3, 4))
            (12, 4, 1)
        """
        ndim = len(dims)
        strides = [1] * ndim
        for i in range(ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * dims[i + 1]
        return tuple(strides)
    
    def linear_index(self, indices: Tuple[int, ...]) -> int:
        """将多维索引转换为线性索引。
        
        Args:
            indices: 多维索引
            
        Returns:
            平坦数组中的线性索引
            
        Raises:
            ValueError: 如果索引长度与 ndim 不匹配
            IndexError: 如果任何索引超出范围
            
        Example:
            >>> shape = Shape((2, 3, 4))
            >>> shape.linear_index((1, 2, 3))
            23
        """
        if len(indices) != self._ndim:
            raise ValueError(f"Expected {self._ndim} indices, got {len(indices)}")
        
        # 检查边界
        for i, (idx, dim) in enumerate(zip(indices, self._dims)):
            if not 0 <= idx < dim:
                raise IndexError(f"Index {idx} out of bounds for dimension {i} with size {dim}")
        
        # 计算线性索引
        linear_idx = 0
        for idx, stride in zip(indices, self._strides):
            linear_idx += idx * stride
        return linear_idx
    
    def can_broadcast(self, other: 'Shape') -> bool:
        """检查该形状是否可以与另一个形状广播。
        
        广播规则（类似 NumPy）：
        1. 从右到左比较维度
        2. 如果维度相等或其中一个为 1，则维度兼容
        3. 缺失的维度视为 1
        
        Args:
            other: 另一个 Shape 对象
            
        Returns:
            如果形状可以广播则返回 True，否则返回 False
            
        Example:
            >>> s1 = Shape((3, 1, 4))
            >>> s2 = Shape((3, 2, 1))
            >>> s1.can_broadcast(s2)
            True
            >>> s3 = Shape((3, 5, 4))
            >>> s1.can_broadcast(s3)
            False
        """
        # 从右到左比较
        for d1, d2 in zip(reversed(self._dims), reversed(other._dims)):
            if d1 != d2 and d1 != 1 and d2 != 1:
                return False
        return True
    
    def broadcast_with(self, other: 'Shape') -> 'Shape':
        """计算与另一个形状广播后的形状。
        
        Args:
            other: 另一个 Shape 对象
            
        Returns:
            表示广播结果的新 Shape
            
        Raises:
            ValueError: 如果形状无法广播
            
        Example:
            >>> s1 = Shape((3, 1, 4))
            >>> s2 = Shape((2, 1))
            >>> result = s1.broadcast_with(s2)
            >>> result.dims
            (3, 2, 4)
        """
        if not self.can_broadcast(other):
            raise ValueError(f"Shapes {self.dims} and {other.dims} cannot be broadcast")
        
        # 在左侧用 1 填充较短的形状
        ndim1, ndim2 = self._ndim, other._ndim
        max_ndim = max(ndim1, ndim2)
        
        dims1 = (1,) * (max_ndim - ndim1) + self._dims
        dims2 = (1,) * (max_ndim - ndim2) + other._dims
        
        # 计算广播维度
        broadcast_dims = []
        for d1, d2 in zip(dims1, dims2):
            broadcast_dims.append(max(d1, d2))
        
        return Shape(tuple(broadcast_dims))
    
    def reshape(self, new_dims: Union[Tuple[int, ...], List[int]]) -> 'Shape':
        """创建一个不同维度但相同大小的新形状。
        
        Args:
            new_dims: 新的维度大小，可以包含一个 -1 用于推断
            
        Returns:
            新的 Shape 对象
            
        Raises:
            ValueError: 如果新形状大小与原始大小不匹配
            
        Example:
            >>> shape = Shape((2, 3, 4))
            >>> new_shape = shape.reshape((6, 4))
            >>> new_shape.dims
            (6, 4)
            >>> new_shape = shape.reshape((-1, 4))
            >>> new_shape.dims
            (6, 4)
        """
        if isinstance(new_dims, list):
            new_dims = tuple(new_dims)
        
        # 处理 -1 用于自动维度推断
        neg_one_count = sum(1 for d in new_dims if d == -1)
        if neg_one_count > 1:
            raise ValueError("Can only specify one unknown dimension as -1")
        
        if neg_one_count == 1:
            known_size = 1
            neg_idx = -1
            for i, d in enumerate(new_dims):
                if d == -1:
                    neg_idx = i
                else:
                    if d <= 0:
                        raise ValueError("Dimensions must be positive integers")
                    known_size *= d
            
            if self._size % known_size != 0:
                raise ValueError(f"Cannot reshape size {self._size} to {new_dims}")
            
            inferred_dim = self._size // known_size
            new_dims = tuple(inferred_dim if i == neg_idx else d 
                           for i, d in enumerate(new_dims))
        
        # Verify size match
        new_size = 1
        for d in new_dims:
            if d <= 0:
                raise ValueError("Dimensions must be positive integers")
            new_size *= d
        
        if new_size != self._size:
            raise ValueError(
                f"Cannot reshape array of size {self._size} to shape {new_dims} "
                f"(size {new_size})"
            )
        
        return Shape(new_dims)
    
    def transpose(self, axes: Tuple[int, ...] = None) -> 'Shape':
        """创建一个维度转置后的新形状。
        
        Args:
            axes: 维度的排列。如果为 None，则反转所有维度
            
        Returns:
            维度转置后的新 Shape 对象
            
        Raises:
            ValueError: 如果 axes 无效
            
        Example:
            >>> shape = Shape((2, 3, 4))
            >>> shape.transpose().dims
            (4, 3, 2)
            >>> shape.transpose((2, 0, 1)).dims
            (4, 2, 3)
        """
        if axes is None:
            # 默认：反转所有维度
            new_dims = tuple(reversed(self._dims))
        else:
            if len(axes) != self._ndim:
                raise ValueError(f"axes length {len(axes)} doesn't match ndim {self._ndim}")
            if set(axes) != set(range(self._ndim)):
                raise ValueError(f"axes must be a permutation of 0..{self._ndim-1}")
            new_dims = tuple(self._dims[i] for i in axes)
        
        return Shape(new_dims)
    
    @property
    def dims(self) -> Tuple[int, ...]:
        """获取维度大小。"""
        return self._dims
    
    @property
    def ndim(self) -> int:
        """获取维度数量。"""
        return self._ndim
    
    @property
    def size(self) -> int:
        """获取元素总数。"""
        return self._size
    
    @property
    def strides(self) -> Tuple[int, ...]:
        """获取每个维度的步长。"""
        return self._strides
    
    def __repr__(self) -> str:
        """Shape 的字符串表示。"""
        return f"Shape{self._dims}"
    
    def __str__(self) -> str:
        """Shape 的字符串表示。"""
        return str(self._dims)
    
    def __eq__(self, other) -> bool:
        """检查与另一个 Shape 的相等性。"""
        if not isinstance(other, Shape):
            return False
        return self._dims == other._dims
    
    def __len__(self) -> int:
        """返回第一个维度的大小（用于兼容性）。"""
        return self._dims[0] if self._dims else 0
    
    def __getitem__(self, idx: int) -> int:
        """通过索引获取维度大小。"""
        return self._dims[idx]
    
    def __iter__(self):
        """迭代维度大小。"""
        return iter(self._dims)
