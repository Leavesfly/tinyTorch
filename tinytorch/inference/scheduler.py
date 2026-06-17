"""连续批处理调度器。"""

import time
from typing import List

from tinytorch.inference.engine import InferenceEngine
from tinytorch.inference.kv_cache import KVCache
from tinytorch.inference.sampling import SamplingParams, sample_token


class Request:
    """一条推理请求（调度单元）。"""

    def __init__(self, req_id: int, prompt: list, params: SamplingParams):
        self.req_id = req_id
        self.prompt = prompt
        self.params = params
        self.generated = list(prompt)
        self.new_tokens = []
        self.kv_cache = None
        self.logits = None
        self.done = False
        self.finish_reason = None
        self.arrival = time.perf_counter()


class ContinuousBatchingScheduler:
    """连续批处理调度器（vLLM 核心思想的教学版）。

    以 token 为粒度调度：每一步为运行中的请求各生成 1 个 token，
    完成的请求立即移出 batch，空位补入等待队列。
    """

    def __init__(self, engine: InferenceEngine, max_batch_size: int = 4):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.finished: List[Request] = []

    def add_request(self, request: Request):
        self.waiting.append(request)

    def _admit(self):
        while self.waiting and len(self.running) < self.max_batch_size:
            req = self.waiting.pop(0)
            req.kv_cache = KVCache(
                self.engine.num_layers,
                self.engine.config.cache_entry_size,
            )
            for _ in req.prompt:
                req.kv_cache.advance()
            req.logits = self.engine.forward_logits(req.generated)
            self.running.append(req)

    def run(self, verbose: bool = True) -> List[Request]:
        """运行调度循环，直到所有请求完成。"""
        step = 0
        while self.waiting or self.running:
            self._admit()
            step += 1

            still_running = []
            for req in self.running:
                next_tok = sample_token(req.logits, req.params, req.generated)
                req.generated.append(next_tok)
                req.new_tokens.append(next_tok)
                req.kv_cache.advance()

                stop = req.params.stop_token
                if stop is not None and next_tok == stop:
                    req.done, req.finish_reason = True, 'stop_token'
                elif len(req.new_tokens) >= req.params.max_tokens:
                    req.done, req.finish_reason = True, 'length'

                if req.done:
                    self.finished.append(req)
                else:
                    req.logits = self.engine.forward_logits(req.generated)
                    still_running.append(req)

            self.running = still_running

            if verbose:
                print(f"    [step {step:2d}] 运行中: {len(self.running)} │ "
                      f"等待: {len(self.waiting)} │ 已完成: {len(self.finished)}")

        return self.finished
