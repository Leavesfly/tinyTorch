"""PyTorch 风格的数据工具。

线程安全说明：
- 本模块中的所有类和函数都是单进程设计，不支持多线程并发访问
- DataLoader 当前仅支持 num_workers=0，不支持多进程数据加载
- 如需多线程安全的数据加载，请使用外部线程安全的数据源
"""

import math
from typing import Iterable, Iterator, List, Mapping, Sequence, Tuple, Optional, Any

from tinytorch.autograd.tensor import Tensor
from tinytorch.ndarr.ndarray import NdArray
from tinytorch.utils import random as tt_random


class Dataset:
    """Map 风格数据集接口。"""

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class IterableDataset:
    """Iterable 风格数据集接口。"""

    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError


class Sampler:
    """所有采样器的基类。"""

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class SequentialSampler(Sampler):
    """顺序采样器。"""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.dataset)))

    def __len__(self) -> int:
        return len(self.dataset)


class RandomSampler(Sampler):
    """随机无放回采样器。"""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.dataset)))
        tt_random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class BatchSampler(Sampler):
    """包装采样器以生成批次索引。"""

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return math.ceil(len(self.sampler) / self.batch_size)


def _unflatten(data: List, dims: Tuple[int, ...]) -> Any:
    if not dims:
        return data[0]
    if len(dims) == 1:
        return data[: dims[0]]
    expected_size = 1
    for dim in dims:
        expected_size *= dim
    if len(data) != expected_size:
        raise ValueError(f"Data size {len(data)} does not match shape {dims} (expected {expected_size})")
    step = len(data) // dims[0]
    return [
        _unflatten(data[i * step : (i + 1) * step], dims[1:])
        for i in range(dims[0])
    ]


def _tensor_to_nested_list(tensor: NdArray) -> List:
    return _unflatten(tensor.data, tensor.shape.dims)


def default_collate(batch: List[Any]) -> Any:
    """默认的 collate 函数，与 PyTorch 行为一致。
    
    将一个样本列表合并为一个批次。支持多种数据类型：
    - NdArray: 堆叠为更高维度的张量
    - Tensor: 递归合并其值并保持梯度信息
    - 数值类型 (int, float): 转换为 NdArray
    - Mapping (dict): 对每个键递归合并对应的值
    - 字符串类型 (str, bytes): 保持为列表
    - namedtuple: 递归合并每个字段
    - Sequence: 转置后递归合并每个元素
    
    Args:
        batch: 样本列表，每个样本可以是任意类型
        
    Returns:
        合并后的批次数据，类型取决于输入类型
        
    Raises:
        ValueError: 当序列类型样本长度不一致时
        
    Example:
        >>> batch = [[1, 2], [3, 4], [5, 6]]
        >>> result = default_collate(batch)
        >>> # 返回 NdArray([[1, 2], [3, 4], [5, 6]])
    """
    if not batch:
        return batch
    elem = batch[0]
    if isinstance(elem, NdArray):
        return NdArray([_tensor_to_nested_list(item) for item in batch], dtype=elem.dtype)
    if isinstance(elem, Tensor):
        values = [item.value for item in batch]
        return Tensor(default_collate(values), requires_grad=elem.requires_grad)
    if isinstance(elem, (int, float)):
        return NdArray(batch)
    if isinstance(elem, Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    if isinstance(elem, (str, bytes)):
        return batch
    if isinstance(elem, tuple) and hasattr(elem, "_fields"):
        return type(elem)(*(default_collate(samples) for samples in zip(*batch)))
    if isinstance(elem, Sequence):
        if not all(len(elem) == len(sample) for sample in batch):
            raise ValueError("All sequences in a batch must have the same size")
        transposed = zip(*batch)
        return [default_collate(list(samples)) for samples in transposed]
    return batch


class DataLoader:
    """PyTorch 风格的数据加载器（单进程）。
    
    线程安全说明：
    - 本类不支持多线程并发访问
    - 迭代器不是线程安全的，不能在多个线程中同时使用
    - 如果需要多线程数据加载，请在外部实现线程安全机制
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler] = None,
        drop_last: bool = False,
        collate_fn: Any = default_collate,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        if num_workers != 0:
            raise NotImplementedError("num_workers > 0 is not supported yet")
        if pin_memory:
            raise NotImplementedError("pin_memory is not supported yet")
        if batch_sampler is not None and (batch_size != 1 or shuffle or sampler is not None or drop_last):
            raise ValueError("batch_sampler is mutually exclusive with batch_size/shuffle/sampler/drop_last")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self._is_iterable = isinstance(dataset, IterableDataset) and not isinstance(dataset, Dataset)
        if self._is_iterable and (shuffle or sampler or batch_sampler):
            raise ValueError("IterableDataset does not support shuffle or samplers")

        if not self._is_iterable:
            if self.batch_sampler is None:
                if self.sampler is None:
                    self.sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
                self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last)

    def __iter__(self) -> Iterator[Any]:
        if self._is_iterable:
            batch = []
            for sample in self.dataset:
                batch.append(sample)
                if len(batch) == self.batch_size:
                    yield self.collate_fn(batch)
                    batch = []
            if batch and not self.drop_last:
                yield self.collate_fn(batch)
            return

        for batch_indices in self.batch_sampler:
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(batch)

    def __len__(self) -> int:
        if self._is_iterable:
            if hasattr(self.dataset, "__len__"):
                if self.drop_last:
                    return len(self.dataset) // self.batch_size
                return math.ceil(len(self.dataset) / self.batch_size)
            raise TypeError("IterableDataset has no length")
        return len(self.batch_sampler)

    def __repr__(self) -> str:
        return (
            f"DataLoader(dataset={self.dataset}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, drop_last={self.drop_last})"
        )
