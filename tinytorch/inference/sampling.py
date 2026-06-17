"""解码采样策略。

提供 greedy / temperature / top-k / top-p / 重复惩罚等自回归生成采样。
"""

import math
import random
from typing import List


class SamplingParams:
    """解码采样参数（对齐 vLLM 的 SamplingParams 设计）。

    Attributes:
        temperature: 温度。0 → 贪心；越大越随机
        top_k: 只在概率最高的 k 个 token 中采样（0 = 不启用）
        top_p: nucleus 采样，只在累计概率达到 p 的最小 token 集合中采样（1.0 = 不启用）
        max_tokens: 最多生成的新 token 数
        stop_token: 遇到该 token 立即停止（EOS）
        repetition_penalty: 重复惩罚（>1 降低已出现 token 的概率）
    """

    def __init__(self, temperature: float = 1.0, top_k: int = 0,
                 top_p: float = 1.0, max_tokens: int = 16,
                 stop_token: int = None, repetition_penalty: float = 1.0):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop_token = stop_token
        self.repetition_penalty = repetition_penalty


def sample_token(logits: List[float], params: SamplingParams,
                 generated: List[int]) -> int:
    """根据采样参数从 logits 中选出下一个 token。

    依次应用：重复惩罚 → 温度 → top-k → top-p → 采样。
    """
    logits = list(logits)

    if params.repetition_penalty != 1.0:
        for tok in set(generated):
            if 0 <= tok < len(logits):
                if logits[tok] > 0:
                    logits[tok] /= params.repetition_penalty
                else:
                    logits[tok] *= params.repetition_penalty

    if params.temperature <= 0:
        return logits.index(max(logits))

    logits = [v / params.temperature for v in logits]

    max_l = max(logits)
    exp_l = [math.exp(v - max_l) for v in logits]
    total = sum(exp_l)
    probs = [v / total for v in exp_l]

    candidates = list(enumerate(probs))

    if params.top_k and params.top_k > 0:
        candidates.sort(key=lambda t: t[1], reverse=True)
        candidates = candidates[:params.top_k]

    if params.top_p < 1.0:
        candidates.sort(key=lambda t: t[1], reverse=True)
        kept = []
        cum = 0.0
        for idx, p in candidates:
            kept.append((idx, p))
            cum += p
            if cum >= params.top_p:
                break
        candidates = kept

    norm = sum(p for _, p in candidates) + 1e-12
    r = random.random()
    cum = 0.0
    for idx, p in candidates:
        cum += p / norm
        if r <= cum:
            return idx
    return candidates[-1][0]
