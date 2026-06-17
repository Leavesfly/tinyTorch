"""KV Cache：自回归推理的增量解码缓存。"""

from typing import List


class KVCache:
    """KV 缓存。

    自回归生成时，第 t 步需要历史所有 token 的 K/V 来做注意力。
    朴素做法每步重算整条序列 → O(n²)；KV Cache 把历史 K/V 存下来，
    每步只算**新 token** 的 K/V 并追加 → O(n)。

    对 MLA 等压缩注意力，可只缓存压缩 latent（如 c_KV），
    需要时再从 latent 恢复完整 K/V。
    """

    def __init__(self, num_layers: int, entry_size: int = 1):
        self.num_layers = num_layers
        self.entry_size = entry_size
        self.cached_entries = [[] for _ in range(num_layers)]
        self.length = 0

    def append(self, layer_idx: int, entry: List[float]):
        """向某层追加一个新 token 的缓存条目。"""
        self.cached_entries[layer_idx].append(entry)

    def get(self, layer_idx: int) -> list:
        return self.cached_entries[layer_idx]

    def advance(self):
        """一个 token 在所有层都处理完后，序列长度 +1。"""
        self.length += 1

    def memory_footprint(self) -> int:
        """返回缓存总维度数（用于显存估算）。"""
        return self.length * self.num_layers * self.entry_size

    def fill_token(self, placeholder: float = 0.0):
        """为当前 token 在所有层写入占位条目并推进长度。"""
        entry = [placeholder] * self.entry_size
        for layer_idx in range(self.num_layers):
            self.append(layer_idx, entry)
        self.advance()
