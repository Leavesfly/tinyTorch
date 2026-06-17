"""推理模块 - 自回归生成与 KV Cache 推理引擎。

该模块提供语言模型推理所需的核心组件：

核心类：
    InferenceEngine: 通用自回归推理引擎（prefill + decode + KV Cache）
    InferenceConfig: 推理配置（层数、缓存维度等）
    SamplingParams: 解码采样参数
    KVCache: 增量解码 KV 缓存
    ContinuousBatchingScheduler: 连续批处理调度器

示例：
    >>> from tinytorch.inference import InferenceEngine, SamplingParams
    >>> engine = InferenceEngine(model)
    >>> out = engine.generate([1, 2, 3], SamplingParams(max_tokens=10))
"""

from tinytorch.inference.sampling import SamplingParams, sample_token
from tinytorch.inference.kv_cache import KVCache
from tinytorch.inference.engine import (
    InferenceConfig,
    InferenceEngine,
    forward_logits,
    infer_inference_config,
)
from tinytorch.inference.scheduler import Request, ContinuousBatchingScheduler

__all__ = [
    'SamplingParams',
    'sample_token',
    'KVCache',
    'InferenceConfig',
    'InferenceEngine',
    'forward_logits',
    'infer_inference_config',
    'Request',
    'ContinuousBatchingScheduler',
]
