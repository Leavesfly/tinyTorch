"""测试数据加载工具。

Author: TinyAI Team
"""

import pytest
from tinytorch.utils.data import (
    Dataset,
    IterableDataset,
    Sampler,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
    DataLoader,
    default_collate,
)
from tinytorch.autograd import Tensor
from tinytorch.ndarr import NdArray


class TestDataset:
    """Dataset 基类的测试。"""

    def test_dataset_getitem(self):
        """验证索引访问。"""
        class MyDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, index):
                return self.data[index]

            def __len__(self):
                return len(self.data)

        dataset = MyDataset([1, 2, 3, 4, 5])
        assert dataset[0] == 1
        assert dataset[2] == 3
        assert dataset[4] == 5

    def test_dataset_len(self):
        """验证长度。"""
        class MyDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, index):
                return self.data[index]

            def __len__(self):
                return len(self.data)

        dataset = MyDataset([1, 2, 3, 4, 5])
        assert len(dataset) == 5


class TestSequentialSampler:
    """SequentialSampler 的测试。"""

    def test_sequential_sampler(self):
        """验证顺序采样。"""
        class MyDataset(Dataset):
            def __init__(self, size):
                self.size = size

            def __getitem__(self, index):
                return index

            def __len__(self):
                return self.size

        dataset = MyDataset(5)
        sampler = SequentialSampler(dataset)
        indices = list(sampler)
        assert indices == [0, 1, 2, 3, 4]

    def test_sequential_sampler_len(self):
        """验证长度。"""
        class MyDataset(Dataset):
            def __init__(self, size):
                self.size = size

            def __getitem__(self, index):
                return index

            def __len__(self):
                return self.size

        dataset = MyDataset(10)
        sampler = SequentialSampler(dataset)
        assert len(sampler) == 10


class TestRandomSampler:
    """RandomSampler 的测试。"""

    def test_random_sampler(self):
        """验证随机采样（所有索引都出现）。"""
        class MyDataset(Dataset):
            def __init__(self, size):
                self.size = size

            def __getitem__(self, index):
                return index

            def __len__(self):
                return self.size

        dataset = MyDataset(5)
        sampler = RandomSampler(dataset)
        indices = list(sampler)
        assert len(indices) == 5
        assert set(indices) == {0, 1, 2, 3, 4}
        # 验证不是顺序的（虽然随机可能偶尔相同，但概率很低）
        assert indices != [0, 1, 2, 3, 4]

    def test_random_sampler_len(self):
        """验证长度。"""
        class MyDataset(Dataset):
            def __init__(self, size):
                self.size = size

            def __getitem__(self, index):
                return index

            def __len__(self):
                return self.size

        dataset = MyDataset(10)
        sampler = RandomSampler(dataset)
        assert len(sampler) == 10


class TestBatchSampler:
    """BatchSampler 的测试。"""

    def test_batch_sampler(self):
        """验证批次生成。"""
        class MyDataset(Dataset):
            def __init__(self, size):
                self.size = size

            def __getitem__(self, index):
                return index

            def __len__(self):
                return self.size

        dataset = MyDataset(10)
        sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=3, drop_last=False)
        batches = list(batch_sampler)
        assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_batch_sampler_drop_last(self):
        """验证 drop_last=True。"""
        class MyDataset(Dataset):
            def __init__(self, size):
                self.size = size

            def __getitem__(self, index):
                return index

            def __len__(self):
                return self.size

        dataset = MyDataset(10)
        sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=3, drop_last=True)
        batches = list(batch_sampler)
        assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def test_batch_sampler_len(self):
        """验证长度计算。"""
        class MyDataset(Dataset):
            def __init__(self, size):
                self.size = size

            def __getitem__(self, index):
                return index

            def __len__(self):
                return self.size

        dataset = MyDataset(10)
        sampler = SequentialSampler(dataset)

        batch_sampler_no_drop = BatchSampler(sampler, batch_size=3, drop_last=False)
        assert len(batch_sampler_no_drop) == 4

        batch_sampler_drop = BatchSampler(sampler, batch_size=3, drop_last=True)
        assert len(batch_sampler_drop) == 3


class TestDataLoader:
    """DataLoader 的测试。"""

    def test_dataloader_basic(self):
        """基本迭代。"""
        class MyDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, index):
                return self.data[index]

            def __len__(self):
                return len(self.data)

        dataset = MyDataset([1, 2, 3, 4, 5])
        dataloader = DataLoader(dataset, batch_size=2)
        batches = list(dataloader)
        assert len(batches) == 3
        assert batches[0].data == [1, 2]
        assert batches[1].data == [3, 4]
        assert batches[2].data == [5]

    def test_dataloader_batch_size(self):
        """验证批次大小。"""
        class MyDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, index):
                return self.data[index]

            def __len__(self):
                return len(self.data)

        dataset = MyDataset(list(range(10)))
        dataloader = DataLoader(dataset, batch_size=3)
        batches = list(dataloader)
        assert len(batches[0].data) == 3
        assert len(batches[1].data) == 3
        assert len(batches[2].data) == 3
        assert len(batches[3].data) == 1

    def test_dataloader_shuffle(self):
        """验证 shuffle 功能。"""
        class MyDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, index):
                return self.data[index]

            def __len__(self):
                return len(self.data)

        dataset = MyDataset(list(range(10)))
        dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
        batches = list(dataloader)
        # 验证数据被打乱了
        first_batch = batches[0].data
        assert first_batch != [0, 1, 2, 3, 4]

    def test_dataloader_drop_last(self):
        """验证 drop_last。"""
        class MyDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, index):
                return self.data[index]

            def __len__(self):
                return len(self.data)

        dataset = MyDataset(list(range(10)))
        dataloader = DataLoader(dataset, batch_size=3, drop_last=True)
        batches = list(dataloader)
        assert len(batches) == 3
        # 每个批次都应该有 3 个元素
        for batch in batches:
            assert len(batch.data) == 3

    def test_dataloader_len(self):
        """验证长度。"""
        class MyDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, index):
                return self.data[index]

            def __len__(self):
                return len(self.data)

        dataset = MyDataset(list(range(10)))
        dataloader_no_drop = DataLoader(dataset, batch_size=3, drop_last=False)
        assert len(dataloader_no_drop) == 4

        dataloader_drop = DataLoader(dataset, batch_size=3, drop_last=True)
        assert len(dataloader_drop) == 3

    def test_dataloader_num_workers_error(self):
        """验证 num_workers>0 抛出 NotImplementedError。"""
        class MyDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, index):
                return self.data[index]

            def __len__(self):
                return len(self.data)

        dataset = MyDataset([1, 2, 3])
        with pytest.raises(NotImplementedError):
            DataLoader(dataset, num_workers=1)


class TestDefaultCollate:
    """default_collate 函数的测试。"""

    def test_collate_tensors(self):
        """测试 Tensor 列表的 collate。"""
        batch = [
            Tensor(NdArray([1.0, 2.0])),
            Tensor(NdArray([3.0, 4.0])),
            Tensor(NdArray([5.0, 6.0])),
        ]
        result = default_collate(batch)
        assert isinstance(result, Tensor)
        assert result.value.shape.dims == (3, 2)
        assert result.value.data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_collate_scalars(self):
        """测试标量列表的 collate。"""
        batch = [1, 2, 3, 4, 5]
        result = default_collate(batch)
        assert isinstance(result, NdArray)
        assert result.shape.dims == (5,)
        assert result.data == [1, 2, 3, 4, 5]

    def test_collate_ndarrays(self):
        """测试 NdArray 列表的 collate。"""
        batch = [
            NdArray([1.0, 2.0]),
            NdArray([3.0, 4.0]),
            NdArray([5.0, 6.0]),
        ]
        result = default_collate(batch)
        assert isinstance(result, NdArray)
        assert result.shape.dims == (3, 2)
        assert result.data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_collate_dict(self):
        """测试字典列表的 collate。"""
        batch = [
            {"x": 1, "y": 2.0},
            {"x": 3, "y": 4.0},
            {"x": 5, "y": 6.0},
        ]
        result = default_collate(batch)
        assert isinstance(result, dict)
        assert "x" in result and "y" in result
        assert result["x"].data == [1, 3, 5]
        assert result["y"].data == [2.0, 4.0, 6.0]

    def test_collate_strings(self):
        """测试字符串列表的 collate。"""
        batch = ["hello", "world", "test"]
        result = default_collate(batch)
        assert result == batch

    def test_collate_nested_list(self):
        """测试嵌套列表的 collate。

        default_collate 对嵌套列表会按列转置后递归 collate，
        结果是每列元素组成的列表。
        """
        batch = [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
        result = default_collate(batch)
        # zip(*batch) 产生 (1,3,5) 和 (2,4,6)，每组再 collate 为 NdArray
        assert isinstance(result, list)
        assert len(result) == 2

    def test_collate_empty_batch(self):
        """测试空批次。"""
        result = default_collate([])
        assert result == []
