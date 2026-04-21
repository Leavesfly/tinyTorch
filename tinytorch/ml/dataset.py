"""数据集类。

提供数据集抽象和批处理功能。

Author: TinyAI Team
"""

from typing import List, Tuple, Optional, Any, Iterator
from tinytorch.ndarr.ndarray import NdArray
from tinytorch.utils.data import Dataset
from tinytorch.utils import random as tt_random


class DataSet(Dataset):
    """数据集类。
    
    封装训练数据和标签，提供批处理、打乱等功能。
    
    Attributes:
        data: 数据样本列表
        labels: 标签列表
        batch_size: 批次大小
        shuffle: 是否在每个 epoch 开始时打乱数据
    
    Example:
        >>> data = [[1, 2], [3, 4], [5, 6], [7, 8]]
        >>> labels = [0, 1, 0, 1]
        >>> dataset = DataSet(data, labels, batch_size=2)
        >>> for batch_data, batch_labels in dataset.get_batches():
        ...     print(batch_data.shape, batch_labels.shape)
    """
    
    def __init__(self, data: List[Any], labels: List[Any], batch_size: int = 32, 
                 shuffle: bool = True) -> None:
        """初始化数据集。
        
        Args:
            data: 数据样本列表
            labels: 标签列表
            batch_size: 批次大小
            shuffle: 是否在每个 epoch 开始时打乱数据
        
        Raises:
            ValueError: 当数据和标签数量不匹配时
        """
        if len(data) != len(labels):
            raise ValueError(
                f"Data and labels must have same length, "
                f"got {len(data)} and {len(labels)}"
            )
        
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = list(range(len(data)))
    
    def __len__(self) -> int:
        """返回数据集大小。"""
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """获取单个样本。
        
        Args:
            index: 样本索引
        
        Returns:
            (data, label) 元组
        """
        return self.data[index], self.labels[index]
    
    def _shuffled_indices(self) -> List[int]:
        """返回一份打乱后的索引副本。"""
        indices = list(range(len(self.data)))
        tt_random.shuffle(indices)
        return indices

    def shuffle_data(self) -> None:
        """打乱数据顺序。"""
        tt_random.shuffle(self._indices)

    def get_batches(self) -> List[Tuple[NdArray, NdArray]]:
        """获取所有批次。
        
        Returns:
            批次列表，每个批次是 (batch_data, batch_labels) 元组
            
        Note:
            对于大数据集，建议使用 __iter__ 方法进行惰性迭代，
            避免一次性将所有批次加载到内存中。
        """
        return list(self.iter_batches())
    
    def _iter_raw_batches(self) -> Iterator[Tuple[List[Any], List[Any]]]:
        """惰性迭代所有批次的原始列表数据（内部公共方法）。

        Yields:
            (batch_data, batch_labels) 元组，均为原始列表。
        """
        if self.shuffle:
            self.shuffle_data()

        num_samples = len(self.data)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_samples)
            batch_indices = self._indices[start_idx:end_idx]

            batch_data = [self.data[idx] for idx in batch_indices]
            batch_labels = [self.labels[idx] for idx in batch_indices]
            yield batch_data, batch_labels

    def iter_batches(self) -> Iterator[Tuple[NdArray, NdArray]]:
        """惰性迭代所有批次，返回 NdArray 格式。

        与 get_batches() 不同，此方法不会一次性创建所有批次，
        而是按需生成，适合大数据集场景。

        Yields:
            (batch_data, batch_labels) 元组，均为 NdArray。
        """
        for batch_data, batch_labels in self._iter_raw_batches():
            yield NdArray(batch_data), NdArray(batch_labels)
    
    def _subset_by_indices(self, indices: List[int], shuffle: bool) -> 'DataSet':
        """根据索引列表创建子数据集。

        Args:
            indices: 样本索引列表
            shuffle: 子数据集是否打乱

        Returns:
            新的 DataSet 实例
        """
        subset_data = [self.data[i] for i in indices]
        subset_labels = [self.labels[i] for i in indices]
        return DataSet(subset_data, subset_labels, batch_size=self.batch_size, shuffle=shuffle)

    def split(self, ratio: float) -> Tuple['DataSet', 'DataSet']:
        """分割数据集。
        
        Args:
            ratio: 第一个数据集的比例（0-1之间）
        
        Returns:
            (train_dataset, val_dataset) 元组
        """
        if not 0 < ratio < 1:
            raise ValueError("Ratio must be between 0 and 1")

        indices = self._shuffled_indices()
        split_idx = int(len(self.data) * ratio)

        train_dataset = self._subset_by_indices(indices[:split_idx], shuffle=self.shuffle)
        val_dataset = self._subset_by_indices(indices[split_idx:], shuffle=False)
        return train_dataset, val_dataset
    
    def __iter__(self) -> Iterator[Tuple[List[Any], List[Any]]]:
        """迭代数据集，产生批次的原始列表数据。

        Yields:
            (batch_data, batch_labels) 元组，均为原始列表。
        """
        return self._iter_raw_batches()
    
    def __repr__(self) -> str:
        """返回数据集的字符串表示。"""
        return (f"DataSet(num_samples={len(self.data)}, "
                f"batch_size={self.batch_size}, shuffle={self.shuffle})")