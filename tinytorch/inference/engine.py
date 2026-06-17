"""推理引擎：prefill + decode 自回归生成。"""

import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from tinytorch.autograd import Tensor
from tinytorch.ndarr import NdArray, Shape

from tinytorch.inference.kv_cache import KVCache
from tinytorch.inference.sampling import SamplingParams, sample_token


@dataclass
class InferenceConfig:
    """推理引擎配置。

    Attributes:
        num_layers: Transformer 层数（决定 KV Cache 层数）
        cache_entry_size: 每层每 token 的压缩缓存维度（如 MLA 的 kv_lora_rank）
        full_kv_entry_size: 传统 MHA 每层每 token 的 K+V 维度（用于显存对比）
    """

    num_layers: int = 1
    cache_entry_size: int = 1
    full_kv_entry_size: int = 2


def infer_inference_config(model: Any) -> InferenceConfig:
    """从常见语言模型结构推断推理配置。

    支持带有 ``num_layers`` / ``layers[0].attention`` 的 Transformer 类模型
    （如 DeepSeek V3 示例）。
    """
    num_layers = getattr(model, 'num_layers', 1)
    cache_entry_size = 1
    full_kv_entry_size = 2

    layers = getattr(model, 'layers', None)
    if layers and len(layers) > 0:
        attn = getattr(layers[0], 'attention', None)
        if attn is not None:
            kv_proj = getattr(attn, 'kv_a_proj', None)
            if kv_proj is not None and hasattr(kv_proj, 'out_features'):
                cache_entry_size = kv_proj.out_features
            attn_dim = getattr(attn, 'attn_dim', None)
            if attn_dim is not None:
                full_kv_entry_size = attn_dim * 2

    return InferenceConfig(
        num_layers=num_layers,
        cache_entry_size=cache_entry_size,
        full_kv_entry_size=full_kv_entry_size,
    )


def forward_logits(model: Any, tokens: List[int]) -> List[float]:
    """将 token 序列送入模型，返回下一个 token 的 logits。"""
    inp = Tensor(
        NdArray([float(i) for i in tokens], Shape((1, len(tokens))), 'float32'),
        requires_grad=False,
    )
    out = model.forward(inp)
    return out.value.data


class InferenceEngine:
    """通用自回归推理引擎。

    封装 prefill（处理 prompt、填充 KV Cache）、decode（逐 token 生成）、
    采样策略与性能指标统计。适用于任何提供 ``forward(input_ids) -> logits``
    接口的生成式模型。

    Example:
        >>> from tinytorch.inference import InferenceEngine, SamplingParams
        >>> engine = InferenceEngine(model)  # 自动推断配置
        >>> result = engine.generate([1, 2, 3], SamplingParams(max_tokens=8))
        >>> result['new_tokens']
    """

    def __init__(self, model: Any, config: Optional[InferenceConfig] = None,
                 forward_fn: Optional[Callable[[Any, List[int]], List[float]]] = None):
        """初始化推理引擎。

        Args:
            model: 生成式模型，需实现 ``forward`` 与 ``eval``
            config: 推理配置；为 None 时尝试从 model 结构自动推断
            forward_fn: 自定义前向函数 ``(model, tokens) -> logits``；
                        默认使用 ``forward_logits``
        """
        self.model = model
        if hasattr(model, 'eval'):
            model.eval()
        self.config = config or infer_inference_config(model)
        self._forward_fn = forward_fn or forward_logits

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    def forward_logits(self, tokens: List[int]) -> List[float]:
        """前向得到下一个 token 的 logits。"""
        return self._forward_fn(self.model, tokens)

    def sample(self, prompt: List[int], max_tokens: int,
               temperature: float = 1.0, **kwargs) -> List[int]:
        """便捷采样接口（PPO rollout 等训练中的生成阶段可用）。"""
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return self.generate(prompt, params)['new_tokens']

    def generate(self, prompt: List[int], params: SamplingParams,
                 verbose: bool = False) -> dict:
        """单条请求生成。

        Returns:
            dict，包含 output / new_tokens / ttft / tpot / total_time /
            finish_reason / cache_dims / full_cache_dims / compression_ratio
        """
        cfg = self.config
        kv_cache = KVCache(cfg.num_layers, cfg.cache_entry_size)
        generated = list(prompt)

        t_start = time.perf_counter()

        for _ in prompt:
            kv_cache.fill_token()

        logits = self.forward_logits(generated)
        ttft = time.perf_counter() - t_start

        new_tokens = []
        finish_reason = 'length'
        decode_start = time.perf_counter()

        for step in range(params.max_tokens):
            next_tok = sample_token(logits, params, generated)
            generated.append(next_tok)
            new_tokens.append(next_tok)
            kv_cache.fill_token()

            if verbose:
                print(f"      step {step + 1:2d}: token={next_tok}")

            if params.stop_token is not None and next_tok == params.stop_token:
                finish_reason = 'stop_token'
                break

            logits = self.forward_logits(generated)

        decode_time = time.perf_counter() - decode_start
        total_time = time.perf_counter() - t_start
        tpot = decode_time / max(len(new_tokens), 1)

        cache_dims = kv_cache.memory_footprint()
        full_cache_dims = kv_cache.length * cfg.num_layers * cfg.full_kv_entry_size

        return {
            'output': generated,
            'new_tokens': new_tokens,
            'ttft': ttft,
            'tpot': tpot,
            'total_time': total_time,
            'finish_reason': finish_reason,
            'cache_dims': cache_dims,
            'full_cache_dims': full_cache_dims,
            'compression_ratio': full_cache_dims / max(cache_dims, 1),
            # 兼容 DeepSeek MLA 示例命名
            'mla_cache_dims': cache_dims,
            'mha_cache_dims': full_cache_dims,
        }
