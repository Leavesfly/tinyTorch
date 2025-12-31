"""数据集类。

提供数据集抽象和批处理功能。

Author: TinyAI Team
"""

import random
from typing import List, Tuple, Optional
from tinytorch.tensor.tensor import Tensor


class DataSet:
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
    
    def __init__(self, data: List, labels: List, batch_size: int = 32, 
                 shuffle: bool = True):
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
    
    def __getitem__(self, index: int) -> Tuple:
        """获取单个样本。
        
        Args:
            index: 样本索引
        
        Returns:
            (data, label) 元组
        """
        return self.data[index], self.labels[index]
    
    def shuffle_data(self) -> None:
        """打乱数据顺序。"""
        random.shuffle(self._indices)
    
    def get_batches(self) -> List[Tuple[Tensor, Tensor]]:
        """获取所有批次。
        
        Returns:
            批次列表，每个批次是 (batch_data, batch_labels) 元组
        """
        # 如果需要打乱，先打乱索引
        if self.shuffle:
            self.shuffle_data()
        
        batches = []
        num_samples = len(self.data)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_samples)
            
            # 获取批次索引
            batch_indices = self._indices[start_idx:end_idx]
            
            # 构建批次数据
            batch_data = [self.data[idx] for idx in batch_indices]
            batch_labels = [self.labels[idx] for idx in batch_indices]
            
            # 转换为 Tensor
            batch_data_tensor = Tensor(batch_data)
            batch_labels_tensor = Tensor(batch_labels)
            
            batches.append((batch_data_tensor, batch_labels_tensor))
        
        return batches
    
    def split(self, ratio: float) -> Tuple['DataSet', 'DataSet']:
        """分割数据集。
        
        Args:
            ratio: 第一个数据集的比例（0-1之间）
        
        Returns:
            (train_dataset, val_dataset) 元组
        """
        if not 0 < ratio < 1:
            raise ValueError("Ratio must be between 0 and 1")
        
        num_samples = len(self.data)
        split_idx = int(num_samples * ratio)
        
        # 先打乱数据
        indices = list(range(num_samples))
        random.shuffle(indices)
        
        # 分割索引
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # 构建训练集和验证集
        train_data = [self.data[i] for i in train_indices]
        train_labels = [self.labels[i] for i in train_indices]
        
        val_data = [self.data[i] for i in val_indices]
        val_labels = [self.labels[i] for i in val_indices]
        
        train_dataset = DataSet(train_data, train_labels, 
                               batch_size=self.batch_size, 
                               shuffle=self.shuffle)
        val_dataset = DataSet(val_data, val_labels,
                             batch_size=self.batch_size,
                             shuffle=False)  # 验证集通常不打乱
        
        return train_dataset, val_dataset
    
    def __repr__(self) -> str:
        """返回数据集的字符串表示。"""
        return (f"DataSet(num_samples={len(self.data)}, "
                f"batch_size={self.batch_size}, shuffle={self.shuffle})")
