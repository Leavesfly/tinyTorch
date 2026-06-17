"""DeepSeek V3 推理引擎演示。

核心推理能力已沉淀为 tinyTorch 基础模块 ``tinytorch.inference``。
本文件演示如何在 DeepSeek V3 模型上使用推理引擎：

  1. KV Cache（增量解码）
  2. 采样策略（greedy / temperature / top-k / top-p）
  3. 性能指标（TTFT / TPOT / 吞吐量）
  4. 连续批处理（Continuous Batching）

Author: TinyAI Team
"""

import random

from tinytorch.inference import (
    ContinuousBatchingScheduler,
    InferenceEngine,
    Request,
    SamplingParams,
)

from examples.deepseek.deepseek_v3_demo import DeepSeekV3Model


def main():
    print("=" * 70)
    print("  DeepSeek V3 推理引擎演示（tinytorch.inference）")
    print("=" * 70)

    random.seed(0)

    model = DeepSeekV3Model(
        vocab_size=50, hidden_size=32, num_layers=2, num_heads=4,
        head_dim=8, kv_lora_rank=16, q_lora_rank=16,
        moe_intermediate_size=64, num_routed_experts=4,
        num_shared_experts=1, top_k=2
    )
    engine = InferenceEngine(model)

    # ── 演示 1：不同采样策略 ──
    print("\n" + "=" * 70)
    print("演示 1 ▶ 采样策略对比（greedy / temperature / top-k / top-p）")
    print("=" * 70)
    prompt = [1, 2, 3]

    configs = {
        'greedy        ': SamplingParams(temperature=0.0, max_tokens=6),
        'temperature   ': SamplingParams(temperature=1.2, max_tokens=6),
        'top-k=5       ': SamplingParams(temperature=1.0, top_k=5, max_tokens=6),
        'top-p=0.9     ': SamplingParams(temperature=1.0, top_p=0.9, max_tokens=6),
        'top-k+rep_pen ': SamplingParams(temperature=1.0, top_k=10,
                                         repetition_penalty=1.3, max_tokens=6),
    }
    print(f"\n  prompt = {prompt}")
    for name, params in configs.items():
        random.seed(0)
        result = engine.generate(prompt, params)
        print(f"  {name}→ 生成: {result['new_tokens']}")

    # ── 演示 2：KV Cache 与 MLA 显存收益 ──
    print("\n" + "=" * 70)
    print("演示 2 ▶ KV Cache 增量解码 & MLA 显存压缩")
    print("=" * 70)
    params = SamplingParams(temperature=0.0, max_tokens=10)
    result = engine.generate(prompt, params, verbose=False)
    print(f"\n  生成 {len(result['new_tokens'])} 个 token: {result['new_tokens']}")
    print(f"\n  KV Cache 显存占用（缓存维度数，越小越省显存）：")
    print(f"    传统 MHA 缓存 K+V : {result['mha_cache_dims']:6d} 维")
    print(f"    DeepSeek MLA 缓存 : {result['mla_cache_dims']:6d} 维（只存压缩 c_KV）")
    print(f"    压缩比            : {result['compression_ratio']:.1f}×")
    print(f"  （真实 V3：32768/512 ≈ 64×，长上下文推理显存大幅下降）")

    # ── 演示 3：性能指标 ──
    print("\n" + "=" * 70)
    print("演示 3 ▶ 推理性能指标（TTFT / TPOT / 吞吐量）")
    print("=" * 70)
    params = SamplingParams(temperature=0.0, max_tokens=12)
    result = engine.generate(prompt, params)
    print(f"\n  TTFT（首 token 延迟）  : {result['ttft'] * 1000:.2f} ms")
    print(f"  TPOT（每 token 时间）  : {result['tpot'] * 1000:.2f} ms")
    print(f"  总耗时                 : {result['total_time'] * 1000:.2f} ms")
    print(f"  吞吐量                 : {len(result['new_tokens']) / result['total_time']:.1f} tokens/s")
    print(f"  停止原因               : {result['finish_reason']}")

    # ── 演示 4：连续批处理 ──
    print("\n" + "=" * 70)
    print("演示 4 ▶ 连续批处理 Continuous Batching（多请求动态调度）")
    print("=" * 70)
    scheduler = ContinuousBatchingScheduler(engine, max_batch_size=2)

    requests = [
        Request(0, [1, 2], SamplingParams(temperature=0.0, max_tokens=3)),
        Request(1, [5, 6, 7], SamplingParams(temperature=0.0, max_tokens=6)),
        Request(2, [10], SamplingParams(temperature=0.0, max_tokens=2)),
        Request(3, [20, 21], SamplingParams(temperature=0.0, max_tokens=5)),
        Request(4, [30, 31, 32], SamplingParams(temperature=0.0, max_tokens=4)),
    ]
    print(f"\n  提交 {len(requests)} 条请求，max_batch_size = 2")
    print(f"  （短请求先完成立即返回，长请求继续，空槽补入新请求）\n")
    for r in requests:
        scheduler.add_request(r)
    finished = scheduler.run(verbose=True)

    print(f"\n  全部完成（按完成顺序）：")
    for req in finished:
        print(f"    请求 {req.req_id}: prompt={req.prompt} → "
              f"生成 {len(req.new_tokens)} token {req.new_tokens} "
              f"［{req.finish_reason}］")

    print("\n" + "=" * 70)
    print("  推理引擎演示结束！（模块：tinytorch.inference）")
    print("=" * 70)


if __name__ == '__main__':
    main()
