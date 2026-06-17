"""推理引擎模块测试。"""

import pytest

from tinytorch.autograd import Tensor
from tinytorch.inference import (
    ContinuousBatchingScheduler,
    InferenceConfig,
    InferenceEngine,
    KVCache,
    Request,
    SamplingParams,
    infer_inference_config,
    sample_token,
)
from tinytorch.ndarr import NdArray, Shape
from tinytorch.nn import Embedding, Linear, Module


class ToyLM(Module):
    """用于测试的最小生成式语言模型。"""

    def __init__(self, vocab_size: int = 10, hidden_size: int = 4, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, hidden_size)
        self.head = Linear(hidden_size, vocab_size, use_bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        emb = self.embedding(input_ids)
        last = emb.value.data[-self.hidden_size:]
        x = Tensor(NdArray(last, Shape((1, self.hidden_size)), 'float32'),
                   requires_grad=False)
        return self.head(x)


def test_sample_token_greedy():
    logits = [0.1, 2.0, 0.5]
    params = SamplingParams(temperature=0.0)
    assert sample_token(logits, params, []) == 1


def test_kv_cache_memory_footprint():
    cache = KVCache(num_layers=2, entry_size=4)
    cache.fill_token()
    cache.fill_token()
    assert cache.length == 2
    assert cache.memory_footprint() == 2 * 2 * 4


def test_infer_inference_config_from_toy_model():
    model = ToyLM(num_layers=3)
    cfg = infer_inference_config(model)
    assert cfg.num_layers == 3


def test_inference_engine_generate():
    model = ToyLM(vocab_size=10, hidden_size=4, num_layers=2)
    config = InferenceConfig(num_layers=2, cache_entry_size=8, full_kv_entry_size=16)
    engine = InferenceEngine(model, config)

    result = engine.generate([1, 2], SamplingParams(temperature=0.0, max_tokens=3))

    assert result['output'] == [1, 2] + result['new_tokens']
    assert len(result['new_tokens']) == 3
    assert result['finish_reason'] == 'length'
    assert result['cache_dims'] > 0
    assert result['compression_ratio'] == pytest.approx(2.0)


def test_inference_engine_sample_helper():
    model = ToyLM()
    engine = InferenceEngine(model, InferenceConfig(num_layers=1))
    tokens = engine.sample([0, 1], max_tokens=2, temperature=0.0)
    assert len(tokens) == 2


def test_stop_token():
    model = ToyLM(vocab_size=10)
    engine = InferenceEngine(model, InferenceConfig(num_layers=1))
    greedy = SamplingParams(temperature=0.0, max_tokens=10, stop_token=0)
    result = engine.generate([1, 2], greedy)
    if 0 in result['new_tokens']:
        assert result['finish_reason'] == 'stop_token'


def test_continuous_batching_scheduler():
    model = ToyLM(vocab_size=10, num_layers=1)
    engine = InferenceEngine(model, InferenceConfig(num_layers=1))
    scheduler = ContinuousBatchingScheduler(engine, max_batch_size=2)

    scheduler.add_request(Request(0, [1], SamplingParams(temperature=0.0, max_tokens=2)))
    scheduler.add_request(Request(1, [2, 3], SamplingParams(temperature=0.0, max_tokens=1)))

    finished = scheduler.run(verbose=False)
    assert len(finished) == 2
    assert all(req.done for req in finished)
