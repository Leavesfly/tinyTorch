"""tinyTorch 的工具模块。"""

from importlib import import_module


_DATA_EXPORTS = {
    "Dataset",
    "IterableDataset",
    "DataLoader",
    "Sampler",
    "RandomSampler",
    "SequentialSampler",
    "BatchSampler",
    "default_collate",
}

__all__ = sorted(_DATA_EXPORTS | {"random"})


def __getattr__(name):
    if name == "random":
        return import_module("tinytorch.utils.random")
    if name in _DATA_EXPORTS:
        data_module = import_module("tinytorch.utils.data")
        return getattr(data_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
