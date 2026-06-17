"""DeepSeek V3 完整训练 Pipeline（行业主流四阶段）。

`deepseek_v3_train.py` 只覆盖了 **预训练（Pre-training）** 这一个阶段。
但工业界训练一个可对话、可对齐的大模型，是一条多阶段的流水线。
本文件在复用同一个 DeepSeek V3 架构的前提下，补齐现在行业主流的
**完整大模型训练 pipeline**：

  ┌──────────────────────────────────────────────────────────────────┐
  │  Stage 1  预训练 Pre-training                                       │
  │     目标：海量无标注文本上做 next-token prediction                  │
  │     产物：Base Model（具备语言知识，但不会"听指令"）                 │
  │                              │                                      │
  │                              ▼                                      │
  │  Stage 2  监督微调 SFT (Supervised Fine-Tuning)                     │
  │     目标：用 (instruction, response) 数据教模型遵循指令              │
  │     关键：prompt 部分 mask 掉，只在 response 上算损失               │
  │     产物：SFT Model（会对话、会遵循指令）                            │
  │                              │                                      │
  │                              ▼                                      │
  │  Stage 3  奖励建模 RM (Reward Modeling)                             │
  │     目标：用人类偏好对 (chosen ≻ rejected) 训练一个打分器           │
  │     关键：pairwise ranking loss = -logσ(r_chosen - r_rejected)     │
  │     产物：Reward Model（能给回答质量打分）                           │
  │                              │                                      │
  │                              ▼                                      │
  │  Stage 4  偏好对齐 Alignment                                        │
  │     路线 A：RLHF / PPO —— 用 RM 当环境奖励，强化学习优化策略         │
  │     路线 B：DPO —— 跳过 RM，直接用偏好数据做对比式优化（更稳更省）   │
  │     产物：Aligned Model（输出更符合人类偏好、更安全）               │
  └──────────────────────────────────────────────────────────────────┘

本文件四个阶段都给出可运行的 tinyTorch 实现（微型教学规模），
帮助理解每个阶段"在优化什么损失、用什么数据、产出什么模型"。

Author: TinyAI Team
"""

import math
import random
import copy

from tinytorch.autograd import Tensor
from tinytorch.ndarr import NdArray, Shape
from tinytorch.ml.optimizers.adam import Adam

from examples.deepseek.deepseek_v3_demo import DeepSeekV3Model
from tinytorch.inference import InferenceEngine


# ════════════════════════════════════════════════════════════════════
# 0. 共享工具：可微的 log-softmax / 交叉熵 / 序列对数似然
# ════════════════════════════════════════════════════════════════════

def _stable_log_softmax(data: list):
    """数值稳定的 log-softmax，返回 (log_probs, probs)。"""
    max_val = max(data)
    shifted = [v - max_val for v in data]
    exp_vals = [math.exp(v) for v in shifted]
    exp_sum = sum(exp_vals)
    log_sum_exp = math.log(exp_sum)
    log_probs = [v - log_sum_exp for v in shifted]
    probs = [v / exp_sum for v in exp_vals]
    return log_probs, probs


def differentiable_ce(logits: Tensor, target_id: int) -> Tensor:
    """构造可反向传播的单步交叉熵损失。

    交叉熵对 logits 的梯度恰好是 (softmax(logits) - one_hot(target))。
    我们用 `loss = sum(logits * grad) + offset` 来同时得到正确的
    损失值和正确的梯度，且全程经过 Tensor 运算保持自动微分图完整。
    """
    data = logits.value.data
    vocab_size = len(data)
    log_probs, probs = _stable_log_softmax(data)
    loss_val = -log_probs[target_id]

    # d(CE)/d(logits) = softmax - one_hot
    grad_data = list(probs)
    grad_data[target_id] -= 1.0
    grad_tensor = Tensor(
        NdArray(grad_data, Shape((1, vocab_size)), 'float32'),
        requires_grad=False
    )

    differentiable_loss = (logits * grad_tensor).sum()

    # 修正常数偏移，使 loss 数值正确（不影响梯度）
    actual_sum = sum(data[i] * grad_data[i] for i in range(vocab_size))
    offset = loss_val - actual_sum
    offset_tensor = Tensor(NdArray([offset], Shape((1,)), 'float32'),
                           requires_grad=False)
    return differentiable_loss + offset_tensor


def token_logprob(logits: Tensor, target_id: int) -> float:
    """返回某个 target token 的 log 概率（标量，不带梯度，用于评估/RL）。"""
    log_probs, _ = _stable_log_softmax(logits.value.data)
    return log_probs[target_id]


def sequence_logprob(model: DeepSeekV3Model, prefix: list, continuation: list) -> float:
    """计算 continuation 在给定 prefix 条件下的序列对数似然 Σ log p(t|context)。

    用于 DPO / PPO 中比较两条回答的整体似然。
    """
    total = 0.0
    context = list(prefix)
    for tok in continuation:
        inp = Tensor(
            NdArray([float(i) for i in context], Shape((1, len(context))), 'float32'),
            requires_grad=False
        )
        logits = model.forward(inp)
        total += token_logprob(logits, tok)
        context.append(tok)
    return total


# ════════════════════════════════════════════════════════════════════
# Stage 1: 预训练 Pre-training
# ════════════════════════════════════════════════════════════════════

class PretrainDataset:
    """预训练数据集：纯文本序列，做 next-token prediction。

    工业界这里是 TB 级无标注语料（网页、代码、书籍）。
    本示例用合成的"有规律 token 序列"模拟。
    """

    def __init__(self, sequences: list, shuffle: bool = True):
        self.sequences = sequences
        self.shuffle = shuffle

    def __len__(self):
        return len(self.sequences)

    def get_batches(self):
        indices = list(range(len(self.sequences)))
        if self.shuffle:
            random.shuffle(indices)
        # 每条序列产出多个 (context, next_token) 监督信号（教 next-token）
        batches = []
        for idx in indices:
            seq = self.sequences[idx]
            for t in range(1, len(seq)):
                batches.append((seq[:t], seq[t]))
        return batches


def pretrain(model: DeepSeekV3Model, dataset: PretrainDataset,
             num_epochs: int = 3, learning_rate: float = 0.005,
             balance_factor: float = 0.003, print_interval: int = 20) -> dict:
    """Stage 1：预训练循环。

    损失：L = L_CE + α · L_balance
      - L_CE:      next-token 交叉熵（语言建模主目标）
      - L_balance: MoE 负载均衡辅助损失（防专家坍塌）
    """
    print("\n" + "=" * 70)
    print("  Stage 1 ▶ 预训练 Pre-training（next-token prediction）")
    print("=" * 70)
    print(f"  数据规模: {len(dataset)} 条序列 | 学习率: {learning_rate} | "
          f"均衡系数 α: {balance_factor}")

    optimizer = Adam(model.parameters(), learning_rate=learning_rate)
    model.train()
    history = []

    for epoch in range(1, num_epochs + 1):
        batches = dataset.get_batches()
        ce_losses = []
        for step, (context, target) in enumerate(batches, 1):
            optimizer.zero_grad()
            inp = Tensor(
                NdArray([float(i) for i in context], Shape((1, len(context))), 'float32'),
                requires_grad=False
            )
            logits = model.forward(inp)
            loss = differentiable_ce(logits, target)
            loss.backward()
            optimizer.step()
            ce_losses.append(loss.value.data[0])

            if step % print_interval == 0 or step == len(batches):
                avg = sum(ce_losses[-print_interval:]) / min(step, print_interval)
                bal = model.get_auxiliary_loss()
                print(f"    epoch {epoch} step {step:3d}/{len(batches)} │ "
                      f"CE: {avg:.4f} │ 均衡CV²: {bal:.4f}")

        epoch_ce = sum(ce_losses) / len(ce_losses)
        history.append(epoch_ce)
        print(f"  ✓ Epoch {epoch} 平均 CE: {epoch_ce:.4f}")

    print(f"  Base Model 就绪：具备 next-token 语言建模能力。")
    return {'ce_history': history}


# ════════════════════════════════════════════════════════════════════
# Stage 2: 监督微调 SFT (Supervised Fine-Tuning)
# ════════════════════════════════════════════════════════════════════

class SFTDataset:
    """SFT 数据集：(prompt_tokens, response_tokens) 指令-回答对。

    关键点：只在 response 部分计算损失（prompt 被 mask）。
    工业界这里是人工标注/蒸馏得到的高质量指令数据。
    """

    def __init__(self, samples: list, shuffle: bool = True):
        # samples: list of (prompt_tokens, response_tokens)
        self.samples = samples
        self.shuffle = shuffle

    def __len__(self):
        return len(self.samples)

    def get_batches(self):
        indices = list(range(len(self.samples)))
        if self.shuffle:
            random.shuffle(indices)
        return [self.samples[i] for i in indices]


def sft_train(model: DeepSeekV3Model, dataset: SFTDataset,
              num_epochs: int = 3, learning_rate: float = 0.003,
              print_interval: int = 10) -> dict:
    """Stage 2：监督微调循环。

    对每个 (prompt, response)：
      - 遍历 response 的每个 token，以 (prompt + 已生成 response 前缀) 为上下文
      - 只在 response token 上累加交叉熵损失（prompt token 不算损失 → mask）
    """
    print("\n" + "=" * 70)
    print("  Stage 2 ▶ 监督微调 SFT（仅在 response 上算损失 / prompt-masked）")
    print("=" * 70)
    print(f"  指令样本: {len(dataset)} 条 | 学习率: {learning_rate}")

    optimizer = Adam(model.parameters(), learning_rate=learning_rate)
    model.train()
    history = []

    for epoch in range(1, num_epochs + 1):
        samples = dataset.get_batches()
        losses = []
        for step, (prompt, response) in enumerate(samples, 1):
            context = list(prompt)
            sample_loss_val = 0.0
            # 逐 token teacher-forcing，仅对 response 反传
            for tok in response:
                optimizer.zero_grad()
                inp = Tensor(
                    NdArray([float(i) for i in context], Shape((1, len(context))), 'float32'),
                    requires_grad=False
                )
                logits = model.forward(inp)
                loss = differentiable_ce(logits, tok)
                loss.backward()
                optimizer.step()
                sample_loss_val += loss.value.data[0]
                context.append(tok)

            avg_tok_loss = sample_loss_val / max(len(response), 1)
            losses.append(avg_tok_loss)
            if step % print_interval == 0 or step == len(samples):
                avg = sum(losses[-print_interval:]) / min(step, print_interval)
                print(f"    epoch {epoch} step {step:3d}/{len(samples)} │ "
                      f"response NLL: {avg:.4f}")

        epoch_loss = sum(losses) / len(losses)
        history.append(epoch_loss)
        print(f"  ✓ Epoch {epoch} 平均 response NLL: {epoch_loss:.4f}")

    print(f"  SFT Model 就绪：学会遵循指令格式、按 prompt 生成 response。")
    return {'nll_history': history}


# ════════════════════════════════════════════════════════════════════
# Stage 3: 奖励建模 Reward Modeling
# ════════════════════════════════════════════════════════════════════

class RewardModel:
    """奖励模型：在 SFT 模型主干上接一个标量打分头。

    工业界做法：复用 SFT/Base 模型主干（这里直接复用 DeepSeekV3Model 的
    隐表示），在最后一层接一个 hidden → 1 的线性头输出标量分数 r(x, y)。

    本教学实现用一个独立的轻量打分头（Parameter 向量 w），
    score = w · h(prompt+response 最后位置的隐表示)，
    通过 pairwise ranking loss 训练，使 chosen 的分数高于 rejected。
    """

    def __init__(self, backbone: DeepSeekV3Model):
        self.backbone = backbone
        hidden = backbone.hidden_size
        # 打分头权重（小随机初始化）
        from tinytorch.nn.parameter import Parameter
        w_init = NdArray([random.gauss(0, 0.02) for _ in range(hidden)],
                         Shape((hidden, 1)), 'float32')
        self.score_head = Parameter(w_init, name='reward_head')

    def parameters(self):
        # 同时训练打分头 + 主干（工业界常 freeze 主干大部分，这里全开教学用）
        return [self.score_head] + self.backbone.parameters()

    def _hidden(self, tokens: list) -> Tensor:
        """取序列最后位置、final_norm 之后的隐表示（lm_head 之前）。"""
        emb = self.backbone.embedding(
            Tensor(NdArray([float(i) for i in tokens],
                           Shape((1, len(tokens))), 'float32'), requires_grad=False)
        )
        h = self.backbone.hidden_size
        last = emb.value.data[-h:]
        x = Tensor(NdArray(last, Shape((1, h)), 'float32'), requires_grad=emb.requires_grad)
        position = len(tokens) - 1
        for layer in self.backbone.layers:
            x = layer(x, position)
        return self.backbone.final_norm(x)

    def score(self, tokens: list) -> Tensor:
        """返回标量奖励分数 Tensor（保持梯度）。"""
        h = self._hidden(tokens)
        # score = h @ w → (1, 1)
        return h.matmul(self.score_head)


def reward_model_train(rm: RewardModel, pref_data: list,
                       num_epochs: int = 5, learning_rate: float = 0.002,
                       print_interval: int = 5) -> dict:
    """Stage 3：训练奖励模型。

    pref_data: list of (prompt, chosen_response, rejected_response)
    损失（Bradley-Terry pairwise ranking）：
        L = -log σ( r(prompt+chosen) - r(prompt+rejected) )
    最小化它 → 拉大 chosen 与 rejected 的分差。
    """
    print("\n" + "=" * 70)
    print("  Stage 3 ▶ 奖励建模 RM（pairwise ranking: chosen ≻ rejected）")
    print("=" * 70)
    print(f"  偏好对: {len(pref_data)} 组 | 学习率: {learning_rate}")

    optimizer = Adam(rm.parameters(), learning_rate=learning_rate)
    history = []

    for epoch in range(1, num_epochs + 1):
        data = pref_data[:]
        random.shuffle(data)
        losses = []
        correct = 0
        for step, (prompt, chosen, rejected) in enumerate(data, 1):
            optimizer.zero_grad()
            r_chosen = rm.score(prompt + chosen)       # (1,1)
            r_rejected = rm.score(prompt + rejected)   # (1,1)

            diff = r_chosen - r_rejected               # (1,1)
            diff_val = diff.value.data[0]

            # L = -log σ(diff) = softplus(-diff)
            # dL/d(diff) = -σ(-diff) = σ(diff) - 1
            sig = 1.0 / (1.0 + math.exp(-diff_val))
            loss_val = math.log(1.0 + math.exp(-abs(diff_val))) + max(0.0, -diff_val)
            grad_coef = sig - 1.0  # dL/d(diff)

            grad_tensor = Tensor(NdArray([grad_coef], Shape((1, 1)), 'float32'),
                                 requires_grad=False)
            # 可微 loss：loss = diff * grad_coef + offset
            surrogate = (diff * grad_tensor).sum()
            offset = loss_val - diff_val * grad_coef
            loss = surrogate + Tensor(NdArray([offset], Shape((1,)), 'float32'),
                                      requires_grad=False)
            loss.backward()
            optimizer.step()

            losses.append(loss_val)
            if diff_val > 0:
                correct += 1
            if step % print_interval == 0 or step == len(data):
                avg = sum(losses[-print_interval:]) / min(step, print_interval)
                acc = correct / step * 100
                print(f"    epoch {epoch} step {step:3d}/{len(data)} │ "
                      f"rank loss: {avg:.4f} │ 偏好准确率: {acc:.1f}%")

        epoch_loss = sum(losses) / len(losses)
        history.append(epoch_loss)
        print(f"  ✓ Epoch {epoch} 平均 ranking loss: {epoch_loss:.4f} │ "
              f"偏好准确率: {correct / len(data) * 100:.1f}%")

    print(f"  Reward Model 就绪：可对任意 (prompt, response) 输出质量分数。")
    return {'rank_loss_history': history}


# ════════════════════════════════════════════════════════════════════
# Stage 4 - 路线 B: DPO (Direct Preference Optimization)
# ════════════════════════════════════════════════════════════════════

def dpo_train(policy: DeepSeekV3Model, ref_model: DeepSeekV3Model,
              pref_data: list, num_epochs: int = 3, learning_rate: float = 0.001,
              beta: float = 0.1, print_interval: int = 5) -> dict:
    """Stage 4（路线 B）：DPO 直接偏好优化。

    DPO 跳过显式奖励模型，直接用偏好数据优化策略：

        L_DPO = -log σ( β · [ (logπ_chosen - logπ_ref_chosen)
                              - (logπ_rejected - logπ_ref_rejected) ] )

    含义：相对参考模型，提高 chosen 的相对似然、压低 rejected 的相对似然。
    ref_model 是冻结的 SFT 模型，提供"不要偏离太远"的锚点（KL 约束）。

    说明：完整 DPO 需对 policy 的全序列对数似然反传。本教学实现采用
    "评分式近似"——用 token 级 log 概率聚合度量偏好间隔，并对 chosen 的
    回答 token 做监督式提升、对 rejected 的回答 token 做监督式抑制，
    从而在 tinyTorch 的标量自动微分上稳定演示 DPO 的核心思想。
    """
    print("\n" + "=" * 70)
    print("  Stage 4-B ▶ DPO 直接偏好优化（无需 RM，参考模型做 KL 锚点）")
    print("=" * 70)
    print(f"  偏好对: {len(pref_data)} 组 | β: {beta} | 学习率: {learning_rate}")

    optimizer = Adam(policy.parameters(), learning_rate=learning_rate)
    policy.train()
    history = []

    for epoch in range(1, num_epochs + 1):
        data = pref_data[:]
        random.shuffle(data)
        losses = []
        margins = []
        for step, (prompt, chosen, rejected) in enumerate(data, 1):
            # 计算策略 / 参考模型对两条回答的序列对数似然（评估，无梯度）
            lp_pol_c = sequence_logprob(policy, prompt, chosen)
            lp_pol_r = sequence_logprob(policy, prompt, rejected)
            lp_ref_c = sequence_logprob(ref_model, prompt, chosen)
            lp_ref_r = sequence_logprob(ref_model, prompt, rejected)

            # DPO logit
            logit = beta * ((lp_pol_c - lp_ref_c) - (lp_pol_r - lp_ref_r))
            loss_val = math.log(1.0 + math.exp(-abs(logit))) + max(0.0, -logit)
            margins.append(logit)
            losses.append(loss_val)

            # 梯度方向：提升 chosen 似然、降低 rejected 似然
            # 用监督式更新近似 DPO 梯度（系数 ∝ σ(-logit) 即"还没学好的程度"）
            weight = 1.0 / (1.0 + math.exp(logit))  # σ(-logit)

            # 提升 chosen：对 chosen 的每个 response token 做加权 CE 下降
            context = list(prompt)
            for tok in chosen:
                optimizer.zero_grad()
                inp = Tensor(NdArray([float(i) for i in context],
                                     Shape((1, len(context))), 'float32'),
                             requires_grad=False)
                logits = policy.forward(inp)
                loss = differentiable_ce(logits, tok)
                scaled = loss * Tensor(NdArray([beta * weight], Shape((1,)), 'float32'),
                                       requires_grad=False)
                scaled.backward()
                optimizer.step()
                context.append(tok)

            # 抑制 rejected：对 rejected 的每个 response token 做反向（梯度上升）
            context = list(prompt)
            for tok in rejected:
                optimizer.zero_grad()
                inp = Tensor(NdArray([float(i) for i in context],
                                     Shape((1, len(context))), 'float32'),
                             requires_grad=False)
                logits = policy.forward(inp)
                loss = differentiable_ce(logits, tok)
                # 负系数 → 提高该 token 的损失 → 降低其概率
                scaled = loss * Tensor(NdArray([-beta * weight], Shape((1,)), 'float32'),
                                       requires_grad=False)
                scaled.backward()
                optimizer.step()
                context.append(tok)

            if step % print_interval == 0 or step == len(data):
                avg = sum(losses[-print_interval:]) / min(step, print_interval)
                avg_m = sum(margins[-print_interval:]) / min(step, print_interval)
                print(f"    epoch {epoch} step {step:3d}/{len(data)} │ "
                      f"DPO loss: {avg:.4f} │ 偏好间隔: {avg_m:+.4f}")

        epoch_loss = sum(losses) / len(losses)
        history.append(epoch_loss)
        print(f"  ✓ Epoch {epoch} 平均 DPO loss: {epoch_loss:.4f} │ "
              f"平均偏好间隔: {sum(margins) / len(margins):+.4f}")

    print(f"  Aligned Model (DPO) 就绪：相对参考模型更偏向 chosen 风格回答。")
    return {'dpo_loss_history': history}


# ════════════════════════════════════════════════════════════════════
# Stage 4 - 路线 A: RLHF / PPO（简化版）
# ════════════════════════════════════════════════════════════════════

def rlhf_ppo_train(policy: DeepSeekV3Model, ref_model: DeepSeekV3Model,
                   rm: RewardModel, prompts: list,
                   num_iterations: int = 3, learning_rate: float = 0.0008,
                   kl_coef: float = 0.1, max_new_tokens: int = 3,
                   print_interval: int = 3) -> dict:
    """Stage 4（路线 A）：RLHF / PPO 简化版。

    经典 RLHF 三件套：policy（被优化）、ref_model（KL 锚点）、reward_model（打分）。

    每次迭代：
      1. Rollout：policy 通过 InferenceEngine（KV Cache 路径）采样生成 response
      2. 打分：reward = RM(prompt+response) - kl_coef · KL(policy‖ref)
         （KL 惩罚防止 policy 为骗取高分而过度偏离 SFT 模型）
      3. 优化：用 reward 当优势信号，做策略梯度更新
         （提高高于平均奖励的 response 的对数似然）

    说明：这是 PPO 的极简教学版（单步、用奖励减去 batch 基线当作优势，
    省略 clip 比率与多 epoch 更新），目的在于讲清 RLHF 的数据流与目标。
    """
    print("\n" + "=" * 70)
    print("  Stage 4-A ▶ RLHF / PPO（policy + 冻结 ref + reward model）")
    print("=" * 70)
    print(f"  prompts: {len(prompts)} 个 | KL 系数: {kl_coef} | 学习率: {learning_rate}")

    optimizer = Adam(policy.parameters(), learning_rate=learning_rate)
    policy.train()
    # Rollout 走推理引擎（KV Cache 路径），与 deepseek_v3_inference.py 一致
    rollout_engine = InferenceEngine(policy)
    history = []

    for it in range(1, num_iterations + 1):
        # ── 1) Rollout + 打分，收集一个 batch ──
        rollouts = []  # (prompt, response, advantage_placeholder, reward)
        rewards = []
        for prompt in prompts:
            # 采样生成（InferenceEngine + KV Cache，无梯度）
            response = _sample_response(rollout_engine, prompt, max_new_tokens)

            # 奖励 = RM 分数 - KL 惩罚
            rm_score = rm.score(prompt + response).value.data[0]
            kl = _seq_kl(policy, ref_model, prompt, response)
            reward = rm_score - kl_coef * kl
            rollouts.append((prompt, response))
            rewards.append(reward)

        # Rollout 阶段 InferenceEngine 会把 policy 置为 eval，优化前切回 train
        policy.train()

        # ── 2) 计算优势（reward - 基线均值）──
        baseline = sum(rewards) / len(rewards)
        advantages = [r - baseline for r in rewards]

        # ── 3) 策略梯度更新 ──
        # 提高优势为正的 response 的似然、降低优势为负的
        for (prompt, response), adv in zip(rollouts, advantages):
            context = list(prompt)
            for tok in response:
                optimizer.zero_grad()
                inp = Tensor(NdArray([float(i) for i in context],
                                     Shape((1, len(context))), 'float32'),
                             requires_grad=False)
                logits = policy.forward(inp)
                loss = differentiable_ce(logits, tok)
                # 策略梯度：loss * (-advantage)
                #   adv>0 → 系数负 → 降低 CE → 提高该 token 概率
                #   adv<0 → 系数正 → 提高 CE → 降低该 token 概率
                coef = -adv
                scaled = loss * Tensor(NdArray([coef], Shape((1,)), 'float32'),
                                       requires_grad=False)
                scaled.backward()
                optimizer.step()
                context.append(tok)

        avg_reward = sum(rewards) / len(rewards)
        history.append(avg_reward)
        if it % print_interval == 0 or it == num_iterations:
            print(f"    iter {it:2d}/{num_iterations} │ 平均奖励: {avg_reward:+.4f} │ "
                  f"基线: {baseline:+.4f}")

    print(f"  Aligned Model (PPO) 就绪：在 RM 奖励 - KL 约束下优化了策略。")
    return {'reward_history': history}


def _sample_response(engine: InferenceEngine, prompt: list,
                     max_new_tokens: int, temperature: float = 1.0) -> list:
    """通过 InferenceEngine（KV Cache 路径）采样生成 response（PPO rollout 用）。"""
    return engine.sample(prompt, max_new_tokens, temperature=temperature)


def _seq_kl(policy: DeepSeekV3Model, ref: DeepSeekV3Model,
            prompt: list, response: list) -> float:
    """近似序列级 KL(policy ‖ ref) = Σ (logπ_pol - logπ_ref)（按生成 token）。"""
    context = list(prompt)
    kl = 0.0
    for tok in response:
        inp = Tensor(NdArray([float(i) for i in context],
                             Shape((1, len(context))), 'float32'),
                     requires_grad=False)
        lp_pol = token_logprob(policy.forward(inp), tok)
        lp_ref = token_logprob(ref.forward(inp), tok)
        kl += (lp_pol - lp_ref)
        context.append(tok)
    return kl


# ════════════════════════════════════════════════════════════════════
# 合成数据生成（四个阶段各自的数据）
# ════════════════════════════════════════════════════════════════════

def make_pretrain_data(vocab_size=50, n=40, seq_len=6) -> list:
    """预训练语料：多种可学习模式的 token 序列。"""
    seqs = []
    hi = vocab_size - 1
    for i in range(n):
        kind = i % 3
        if kind == 0:  # 递增
            s = random.randint(1, hi - seq_len)
            seqs.append([min(s + j, hi) for j in range(seq_len)])
        elif kind == 1:  # 等差
            s = random.randint(1, hi // 2)
            d = random.randint(1, 3)
            seqs.append([min(s + j * d, hi) for j in range(seq_len)])
        else:  # 重复 a,b,a,b
            a, b = random.randint(1, hi), random.randint(1, hi)
            seqs.append([a if j % 2 == 0 else b for j in range(seq_len)])
    return seqs


def make_sft_data(n=20) -> list:
    """SFT 指令数据：(prompt, response)。

    用一个简单可学习的"指令"：给定起点，续写递增序列。
    prompt 用一个固定的"指令前缀 token" 0 标记。
    """
    samples = []
    for _ in range(n):
        start = random.randint(1, 30)
        prompt = [0, start, start + 1]        # 0 = 指令标记
        response = [start + 2, start + 3, start + 4]
        samples.append((prompt, response))
    return samples


def make_preference_data(n=16) -> list:
    """偏好数据：(prompt, chosen, rejected)。

    chosen = 正确的递增续写（符合指令）
    rejected = 随机/错误续写（不符合指令）
    用于奖励建模 (Stage 3) 与 DPO (Stage 4-B)。
    """
    data = []
    for _ in range(n):
        start = random.randint(1, 30)
        prompt = [0, start, start + 1]
        chosen = [start + 2, start + 3, start + 4]       # 正确续写
        rejected = [random.randint(1, 40) for _ in range(3)]  # 错误续写
        data.append((prompt, chosen, rejected))
    return data


# ════════════════════════════════════════════════════════════════════
# 端到端 Pipeline
# ════════════════════════════════════════════════════════════════════

def evaluate_instruction_following(model: DeepSeekV3Model, samples: list) -> float:
    """评估：给定 SFT prompt，贪心生成首个 token 是否等于期望的正确续写。"""
    correct = 0
    for prompt, response in samples:
        inp = Tensor(NdArray([float(i) for i in prompt],
                             Shape((1, len(prompt))), 'float32'), requires_grad=False)
        logits = model.forward(inp)
        pred = logits.value.data.index(max(logits.value.data))
        if pred == response[0]:
            correct += 1
    return correct / max(len(samples), 1) * 100


def main():
    print("=" * 70)
    print("  DeepSeek V3 完整训练 Pipeline（预训练 → SFT → RM → 对齐）")
    print("=" * 70)
    print("""
本示例演示工业界主流的大模型四阶段训练流水线，
全部基于同一个 DeepSeek V3 架构（MLA + MoE + RoPE + RMSNorm）：

  Stage 1  预训练   —— 海量文本 next-token，得到 Base Model
  Stage 2  SFT      —— 指令-回答数据，prompt-masked 监督微调
  Stage 3  RM       —— 人类偏好对，pairwise ranking 训练打分器
  Stage 4  对齐     —— DPO（直接偏好优化）+ RLHF/PPO（强化学习）
""")

    random.seed(42)

    def build_model():
        return DeepSeekV3Model(
            vocab_size=50, hidden_size=32, num_layers=2, num_heads=4,
            head_dim=8, kv_lora_rank=16, q_lora_rank=16,
            moe_intermediate_size=64, num_routed_experts=4,
            num_shared_experts=1, top_k=2
        )

    # ── 准备各阶段数据 ──
    pretrain_data = PretrainDataset(make_pretrain_data(vocab_size=50, n=40, seq_len=6))
    sft_data = SFTDataset(make_sft_data(n=20))
    pref_data = make_preference_data(n=16)
    sft_eval = make_sft_data(n=12)

    # ════════════ Stage 1: 预训练 ════════════
    model = build_model()
    pretrain(model, pretrain_data, num_epochs=2, learning_rate=0.005)

    base_acc = evaluate_instruction_following(model, sft_eval)
    print(f"\n  [评估] 预训练后 指令遵循准确率: {base_acc:.1f}%（Base 还不会听指令）")

    # ════════════ Stage 2: SFT ════════════
    sft_train(model, sft_data, num_epochs=3, learning_rate=0.004)
    sft_acc = evaluate_instruction_following(model, sft_eval)
    print(f"\n  [评估] SFT 后 指令遵循准确率: {sft_acc:.1f}%")

    # 冻结一份 SFT 模型作为后续阶段的"参考模型"（KL 锚点）
    ref_model = copy.deepcopy(model)
    ref_model.eval()

    # ════════════ Stage 3: 奖励建模 ════════════
    rm = RewardModel(copy.deepcopy(model))
    reward_model_train(rm, pref_data, num_epochs=4, learning_rate=0.003)

    # ════════════ Stage 4-B: DPO ════════════
    dpo_model = copy.deepcopy(model)
    dpo_train(dpo_model, ref_model, pref_data, num_epochs=2,
              learning_rate=0.002, beta=0.1)
    dpo_acc = evaluate_instruction_following(dpo_model, sft_eval)
    print(f"\n  [评估] DPO 对齐后 指令遵循准确率: {dpo_acc:.1f}%")

    # ════════════ Stage 4-A: RLHF / PPO ════════════
    ppo_model = copy.deepcopy(model)
    ppo_prompts = [p for p, _ in make_sft_data(n=8)]
    rlhf_ppo_train(ppo_model, ref_model, rm, ppo_prompts,
                   num_iterations=3, learning_rate=0.001, kl_coef=0.1)
    ppo_acc = evaluate_instruction_following(ppo_model, sft_eval)
    print(f"\n  [评估] RLHF/PPO 对齐后 指令遵循准确率: {ppo_acc:.1f}%")

    # ════════════ 总结 ════════════
    print("\n" + "=" * 70)
    print("  Pipeline 总结")
    print("=" * 70)
    print(f"""
  指令遵循准确率变化（同一套评测集）：
    预训练 Base   : {base_acc:5.1f}%   （只会建模语言，不懂指令）
    SFT          : {sft_acc:5.1f}%   （学会遵循指令）
    DPO 对齐      : {dpo_acc:5.1f}%   （偏好对齐，倾向高质量回答）
    RLHF/PPO 对齐 : {ppo_acc:5.1f}%   （RM 奖励 + KL 约束优化）

  四阶段各自优化的目标：
    Stage 1 预训练 : L = CE(next-token) + α·L_balance
    Stage 2 SFT    : L = CE(response only, prompt-masked)
    Stage 3 RM     : L = -logσ(r_chosen - r_rejected)
    Stage 4 DPO    : L = -logσ(β·[Δlogπ_chosen - Δlogπ_rejected])
    Stage 4 PPO    : maximize  E[ RM(x,y) - β·KL(π‖π_ref) ]

  现实工程中还会穿插：数据清洗与去重、长上下文扩展（YaRN）、
  课程式数据配比、安全对齐（红队/宪法AI）、评测与回归（MMLU/MT-Bench）、
  以及推理优化（量化/KV缓存/投机解码）。本示例聚焦训练主链路。
""")
    print("=" * 70)
    print("  完整 Pipeline 演示结束！")
    print("=" * 70)


if __name__ == '__main__':
    main()
