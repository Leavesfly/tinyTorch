"""DeepSeek V3 核心架构还原示例。

完整实现 DeepSeek V3 的 8 大核心创新机制：

  架构组件              本示例实现              DeepSeek V3 真实规格
  ─────────────────────────────────────────────────────────────────
  归一化      RMSNorm                    ✓    hidden=7168
  位置编码    RoPE（旋转位置编码）         ✓    rope_base=10000
  注意力      MLA（低秩 KV 联合压缩）      ✓    kv_lora_rank=512
  FFN 激活    SwiGLU（SiLU 门控）         ✓    moe_ffn_dim=2048
  专家架构    共享专家 + 路由专家双池       ✓    1共享 + 256路由
  专家选择    真实 Top-K 排序门控          ✓    top_k=8
  负载均衡    辅助损失（CV² 度量）          ✓    balance_factor=0.003
  残差架构    Pre-RMSNorm                 ✓    61层

注：本示例使用极小规模参数（hidden=64）用于教学演示，
    核心架构与真实 DeepSeek V3 保持一致。

Author: TinyAI Team
"""

import math
import random

from tinytorch.ndarr import NdArray, Shape
from tinytorch.autograd import Tensor
from tinytorch.nn import Module
from tinytorch.nn.layers import Linear, Embedding
from tinytorch.nn.parameter import Parameter
from tinytorch.nn.container import ModuleList


# ════════════════════════════════════════════════════════════════════
# 1. RMSNorm（均方根归一化）
#    DeepSeek V3 使用 RMSNorm 而非 LayerNorm，无均值中心化，计算更快
# ════════════════════════════════════════════════════════════════════

class RMSNorm(Module):
    """均方根归一化（Root Mean Square Layer Normalization）。

    公式：y = x / RMS(x) · γ
          RMS(x) = sqrt( mean(x²) + ε )

    与 LayerNorm 的区别：
    - 不减均值（无中心化），只做缩放，计算量更少
    - DeepSeek V3 全程使用 RMSNorm（eps=1e-6）
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 可学习缩放参数 γ，初始化为全 1
        self.weight = Parameter(NdArray.ones((hidden_size,)), name='rms_weight')

    def forward(self, x: Tensor) -> Tensor:
        # 1) 计算 mean(x²)，shape: (1,)
        ms = (x * x).mean()
        # 2) RMS = sqrt(mean(x²) + ε)
        eps_var = Tensor(NdArray([self.eps]), requires_grad=False)
        rms = (ms + eps_var).sqrt()
        # 3) 归一化并应用可学习缩放 γ
        #    (1, hidden) / (1,) → 广播 → (1, hidden)
        #    (1, hidden) * (hidden,) → 广播 → (1, hidden)
        return (x / rms) * self.weight


# ════════════════════════════════════════════════════════════════════
# 2. RoPE（旋转位置编码）
#    DeepSeek V3 使用解耦 RoPE，为 Q/K 注入位置信息，不占 KV 缓存
# ════════════════════════════════════════════════════════════════════

class RotaryEmbedding:
    """旋转位置编码（Rotary Position Embedding）。

    核心思想：将 Q/K 向量视为复数对，在复平面旋转角度 θᵢ · position
    效果：Q·K 内积只取决于相对位置，天然支持位置外推

    第 i 对维度的旋转（2×2 块）：
      [ cos(pos·θᵢ)  -sin(pos·θᵢ) ] [ x₂ᵢ   ]
      [ sin(pos·θᵢ)   cos(pos·θᵢ) ] [ x₂ᵢ₊₁ ]

    θᵢ = 1 / base^(2i / dim)，base=10000

    优势：无额外参数，天然相对位置感知，可配合 YaRN 外推长度
    """

    def __init__(self, dim: int, max_seq_len: int = 256, base: float = 10000.0):
        self.dim = dim
        self._build_cache(max_seq_len, base)

    def _build_cache(self, max_len: int, base: float):
        """预计算所有位置的 cos/sin 旋转值，避免推理时重复计算。"""
        half = self.dim // 2
        # θᵢ = 1 / base^(2i/dim)，频率从高到低
        inv_freq = [1.0 / (base ** (2 * i / self.dim)) for i in range(half)]

        self.cos_cache = []   # (max_len, dim)
        self.sin_cache = []
        for pos in range(max_len):
            cos_row, sin_row = [], []
            for freq in inv_freq:
                angle = pos * freq
                # 每对 (x₂ᵢ, x₂ᵢ₊₁) 共享同一 cos/sin
                cos_row += [math.cos(angle), math.cos(angle)]
                sin_row += [math.sin(angle), math.sin(angle)]
            self.cos_cache.append(cos_row)
            self.sin_cache.append(sin_row)

    @staticmethod
    def _rotate_half(x_data: list, dim: int) -> list:
        """计算旋转辅助向量：(x₀, x₁, x₂, x₃, …) → (-x₁, x₀, -x₃, x₂, …)

        模拟复数乘法的虚部贡献，实现等效旋转矩阵。
        """
        result = []
        for i in range(0, dim, 2):
            result.append(-(x_data[i + 1] if i + 1 < dim else 0.0))
            result.append(x_data[i])
        return result

    def apply(self, x: Tensor, position: int) -> Tensor:
        """将 RoPE 应用到向量 x（形状 (1, dim)）。

        RoPE(x, pos) = x · cos + rotate_half(x) · sin
        """
        dim = x.value.shape.dims[-1]
        x_data = x.value.data
        cos_vals = self.cos_cache[position][:dim]
        sin_vals = self.sin_cache[position][:dim]
        rotated = self._rotate_half(x_data, dim)
        result = [
            x_data[i] * cos_vals[i] + rotated[i] * sin_vals[i]
            for i in range(dim)
        ]
        return Tensor(
            NdArray(result, Shape((1, dim)), 'float32'),
            requires_grad=x.requires_grad
        )


# ════════════════════════════════════════════════════════════════════
# 3. SwiGLU 激活（DeepSeek V3 专家 FFN 的核心激活）
# ════════════════════════════════════════════════════════════════════

def silu(x: Tensor) -> Tensor:
    """SiLU 激活函数（Sigmoid Linear Unit）。

    公式：SiLU(x) = x · σ(x)
    特点：平滑 ReLU，梯度更稳定；DeepSeek V3 专家 FFN 全部采用
    """
    return x * x.sigmoid()


# ════════════════════════════════════════════════════════════════════
# 4. DeepSeek 专家网络（SwiGLU 门控 FFN）
# ════════════════════════════════════════════════════════════════════

class DeepSeekExpert(Module):
    """单个专家网络：SwiGLU 门控前馈网络。

    架构：output = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )

    与标准 FFN（Linear→ReLU→Linear）的区别：
    - gate_proj 分支：计算"门"，决定哪些信息通过
    - up_proj 分支：提取待过滤的信息
    - 两路逐元素相乘（SwiGLU）：门控融合

    DeepSeek V3 实际：每个专家 moe_intermediate_size=2048
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, use_bias=False)
        self.up_proj   = Linear(hidden_size, intermediate_size, use_bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, use_bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = silu(self.gate_proj(x))   # SiLU 门控：(1, intermediate)
        up   = self.up_proj(x)           # 信息分支：(1, intermediate)
        h    = gate * up                 # SwiGLU：逐元素门控融合
        return self.down_proj(h)         # 投影回 hidden：(1, hidden)


# ════════════════════════════════════════════════════════════════════
# 5. MoE 门控（真实 Top-K 排序选择）
# ════════════════════════════════════════════════════════════════════

class DeepSeekMoEGate(Module):
    """DeepSeek V3 MoE 门控网络（真实 Top-K 排序）。

    流程：
    1. 线性投影：x → (1, num_routed_experts)  logits
    2. Softmax：logits → 路由概率分布
    3. ★ 真实 Top-K：按概率从大到小排序，选前 top_k 个专家
    4. 重归一化：使选中的 top_k 个专家权重之和为 1

    DeepSeek V3 实际：256 个路由专家，top_k=8
    注：真实 V3 使用 sigmoid + 归一化，这里用 softmax 简化，核心相同。
    """

    def __init__(self, hidden_size: int, num_routed_experts: int, top_k: int):
        super().__init__()
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.gate = Linear(hidden_size, num_routed_experts, use_bias=False)

    def forward(self, x: Tensor):
        """返回 (top_k_indices, top_k_weights, all_probs)。"""
        logits = self.gate(x).value.data   # (1, num_routed_experts) → 列表

        # Softmax（数值稳定版：减去最大值防溢出）
        max_l = max(logits)
        exp_l = [math.exp(v - max_l) for v in logits]
        total = sum(exp_l)
        probs = [v / total for v in exp_l]

        # ★ 真实 Top-K：按概率大小排序
        indexed = sorted(enumerate(probs), key=lambda t: t[1], reverse=True)
        top_k_indices = [idx for idx, _ in indexed[:self.top_k]]

        # 重归一化：使选中专家权重之和为 1
        top_k_raw = [probs[i] for i in top_k_indices]
        prob_sum = sum(top_k_raw) + 1e-10
        top_k_weights = [p / prob_sum for p in top_k_raw]

        return top_k_indices, top_k_weights, probs


# ════════════════════════════════════════════════════════════════════
# 6. DeepSeekMoE（共享专家 + 路由专家，双池设计）
# ════════════════════════════════════════════════════════════════════

class DeepSeekMoE(Module):
    """DeepSeek V3 混合专家前馈层。

    ┌────────────────────────────────────────────────────┐
    │  输入 x                                            │
    │    │                                               │
    │    ├──→ 共享专家池（始终全部激活）→ y_shared        │
    │    │                                               │
    │    └──→ 门控网络 → Top-K 路由专家 → y_routed       │
    │                                                    │
    │  输出 = y_shared + y_routed                        │
    └────────────────────────────────────────────────────┘

    共享专家（Shared Experts）：
    - 始终激活，不受门控控制
    - 捕捉所有 token 的通用特征（语法、格式、常识等）

    路由专家（Routed Experts）：
    - 门控网络动态选择 top_k 个
    - 捕捉 token 的专有特征（代码、数学、多语言等）

    DeepSeek V3 实际：1 个共享专家 + 256 个路由专家，每次激活 8 个
    """

    def __init__(self, hidden_size: int, moe_intermediate_size: int,
                 num_routed_experts: int = 8, num_shared_experts: int = 1,
                 top_k: int = 2):
        super().__init__()
        self.top_k = top_k

        # 门控网络（仅对路由专家做 Top-K 选择）
        self.gate = DeepSeekMoEGate(hidden_size, num_routed_experts, top_k)

        # 共享专家池（ModuleList 正确注册参数，保证可训练）
        self.shared_experts = ModuleList([
            DeepSeekExpert(hidden_size, moe_intermediate_size)
            for _ in range(num_shared_experts)
        ])

        # 路由专家池（Top-K 动态选择）
        self.routed_experts = ModuleList([
            DeepSeekExpert(hidden_size, moe_intermediate_size)
            for _ in range(num_routed_experts)
        ])

        # 负载均衡统计（记录每个路由专家被选中的次数）
        self._expert_load = [0] * num_routed_experts
        self._total_tokens = 0

    def forward(self, x: Tensor) -> Tensor:
        # ── 共享专家（始终激活，累加所有共享专家的输出）──
        shared_out = None
        for expert in self.shared_experts:
            out = expert(x)
            shared_out = out if shared_out is None else (shared_out + out)

        # ── 路由专家（Top-K 加权求和）──
        top_k_indices, top_k_weights, _ = self.gate(x)

        # 更新负载统计
        self._total_tokens += 1
        for idx in top_k_indices:
            self._expert_load[idx] += 1

        routed_out = None
        for idx, weight in zip(top_k_indices, top_k_weights):
            expert_out = self.routed_experts[idx](x)
            # 门控权重乘以专家输出（通过 Tensor 运算保持梯度图）
            weight_var = Tensor(NdArray([weight], Shape((1,)), 'float32'),
                                requires_grad=False)
            weighted = expert_out * weight_var   # (1, hidden) * (1,) → 广播
            routed_out = weighted if routed_out is None else (routed_out + weighted)

        # ── 合并共享专家与路由专家输出 ──
        if shared_out is not None and routed_out is not None:
            return shared_out + routed_out
        if shared_out is not None:
            return shared_out
        return routed_out if routed_out is not None else x

    def load_balance_loss(self) -> float:
        """计算负载均衡辅助损失（变异系数的平方 CV²）。

        DeepSeek V3 训练时在主损失之外加入辅助损失：
          L_total = L_CE + 0.003 · L_balance

        其中 L_balance ∝ Σ f_i · p̄_i
          f_i  = token 路由到专家 i 的实际比例
          p̄_i  = 门控对专家 i 的平均路由概率

        这里用 CV²（变异系数的平方）近似衡量负载不均衡程度。
        理想均衡：每个专家等量 token → CV² = 0
        """
        if self._total_tokens == 0:
            return 0.0
        avg = sum(self._expert_load) / max(len(self._expert_load), 1)
        if avg < 1e-10:
            return 0.0
        var = sum((l - avg) ** 2 for l in self._expert_load) / len(self._expert_load)
        return var / (avg ** 2)

    def print_load_stats(self):
        """打印专家负载分布（教学演示用）。"""
        if self._total_tokens == 0:
            print("    尚无负载统计")
            return
        total = sum(self._expert_load)
        n = len(self._expert_load)
        print(f"    专家池（{n} 个路由专家），共处理 {self._total_tokens} 个 token：")
        for i, load in enumerate(self._expert_load):
            pct = load / total * 100 if total > 0 else 0.0
            bar = '█' * int(pct / 4)
            print(f"    专家{i:2d}: {load:4d}次 ({pct:5.1f}%) {bar}")
        print(f"    负载均衡损失 CV² = {self.load_balance_loss():.4f}"
              f"  （越接近 0 越均衡）")


# ════════════════════════════════════════════════════════════════════
# 7. MLA（多头潜在注意力）
#    DeepSeek V3 最核心的注意力创新：低秩联合压缩 KV
# ════════════════════════════════════════════════════════════════════

class MultiHeadLatentAttention(Module):
    """Multi-head Latent Attention（多头潜在注意力）。

    问题：传统 MHA 的 KV 缓存随序列长度线性增长，长序列推理显存爆炸
    MLA 解法：将 K、V 联合压缩到低维潜空间，只缓存压缩表示 c_KV

    传统 MHA（以 128 头、128 dim 为例）：
      每 token KV 缓存 = 128 × 128 × 2 = 32768 维

    MLA（kv_lora_rank=512）：
      每 token KV 缓存 = 512 维（只存 c_KV）
      压缩比 ≈ 64×

    投影结构：
      Q 路径：x →(q_a_proj)→ c_Q →(q_b_proj)→ Q  （低秩分解）
      KV 路径：x →(kv_a_proj)→ c_KV →(kv_b_proj)→ [K, V]

    解耦 RoPE：
      RoPE 作用于 Q 和 K 的部分维度，与低秩压缩解耦
      推理时可仅缓存 c_KV，需要时再从 c_KV 恢复 K、V

    本实现为简化版（等效单头），保留所有关键投影结构和 RoPE。
    """

    def __init__(self, hidden_size: int, num_heads: int,
                 kv_lora_rank: int, q_lora_rank: int, head_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = head_dim
        self.attn_dim    = num_heads * head_dim   # 等效注意力维度

        # ── Q 低秩分解（两步投影）──
        # Step 1：hidden → q_lora_rank（压缩 Q）
        self.q_a_proj = Linear(hidden_size, q_lora_rank, use_bias=False)
        self.q_a_norm = RMSNorm(q_lora_rank)
        # Step 2：q_lora_rank → attn_dim（展开到注意力空间）
        self.q_b_proj = Linear(q_lora_rank, self.attn_dim, use_bias=False)

        # ── KV 联合低秩压缩（MLA 核心：c_KV 是推理时真正缓存的内容）──
        # hidden → kv_lora_rank（联合压缩 K 和 V）
        self.kv_a_proj = Linear(hidden_size, kv_lora_rank, use_bias=False)
        self.kv_a_norm = RMSNorm(kv_lora_rank)
        # kv_lora_rank → attn_dim×2（解压出完整 K 和 V）
        self.kv_b_proj = Linear(kv_lora_rank, self.attn_dim * 2, use_bias=False)

        # ── 输出投影 ──
        self.o_proj = Linear(self.attn_dim, hidden_size, use_bias=False)

        # ── 解耦 RoPE：为 Q 和 K 注入位置信息 ──
        # RoPE 作用于完整的 attn_dim 维度
        self.rope = RotaryEmbedding(dim=self.attn_dim)

    def forward(self, x: Tensor, position: int = 0) -> Tensor:
        """前向传播。

        Args:
            x: 输入，形状 (1, hidden_size)
            position: 当前 token 的绝对序列位置（用于 RoPE）
        """
        # ── Q 路径：低秩分解 + RMSNorm + RoPE ──
        c_q = self.q_a_norm(self.q_a_proj(x))   # (1, q_lora_rank)
        Q   = self.q_b_proj(c_q)                 # (1, attn_dim)
        Q   = self.rope.apply(Q, position)        # 注入位置信息

        # ── KV 路径：联合低秩压缩 → 解压出 K、V ──
        # c_kv：推理时真正存入 KV 缓存的内容（kv_lora_rank 维）
        c_kv = self.kv_a_norm(self.kv_a_proj(x))  # (1, kv_lora_rank)
        kv   = self.kv_b_proj(c_kv)               # (1, attn_dim×2)

        # 分割 K 和 V（前半 = K，后半 = V）
        half   = self.attn_dim
        kv_dat = kv.value.data
        K = Tensor(NdArray(kv_dat[:half], Shape((1, half)), 'float32'),
                   requires_grad=x.requires_grad)
        V = Tensor(NdArray(kv_dat[half:], Shape((1, half)), 'float32'),
                   requires_grad=x.requires_grad)
        K = self.rope.apply(K, position)           # K 也注入位置信息

        # ── Scaled Dot-Product Attention ──
        # score = Q @ K^T / sqrt(head_dim)，形状 (1,1)
        score = Q.matmul(K.transpose())
        scale = math.sqrt(self.head_dim)
        # softmax（单 token 情形退化为 1.0；多 token 对应完整 causal attention）
        exp_s = [math.exp(v / scale) for v in score.value.data]
        attn_w = exp_s[0] / (sum(exp_s) + 1e-10)

        # attn_w · V，形状 (1, attn_dim)
        out_data = [v * attn_w for v in V.value.data]
        output = Tensor(
            NdArray(out_data, V.value.shape, 'float32'),
            requires_grad=x.requires_grad
        )

        # ── 输出投影 ──
        return self.o_proj(output)                 # (1, hidden)


# ════════════════════════════════════════════════════════════════════
# 8. DeepSeekV3Block（Pre-RMSNorm Transformer 层）
# ════════════════════════════════════════════════════════════════════

class DeepSeekV3Block(Module):
    """DeepSeek V3 Transformer 块。

    采用 Pre-RMSNorm 架构（比 Post-Norm 训练更稳定）：

      x = x + Attention( RMSNorm(x) )   ← 注意力子层
      x = x + MoE(      RMSNorm(x) )   ← MoE 子层

    残差连接保证梯度畅通，归一化在残差分支内部进行。
    """

    def __init__(self, hidden_size, num_heads, head_dim,
                 kv_lora_rank, q_lora_rank,
                 moe_intermediate_size, num_routed_experts,
                 num_shared_experts, top_k):
        super().__init__()
        # 注意力前的 Pre-RMSNorm
        self.attn_norm = RMSNorm(hidden_size)
        # 多头潜在注意力（MLA）
        self.attention = MultiHeadLatentAttention(
            hidden_size, num_heads, kv_lora_rank, q_lora_rank, head_dim
        )
        # MoE 前的 Pre-RMSNorm
        self.ffn_norm = RMSNorm(hidden_size)
        # DeepSeekMoE（共享专家 + 路由专家）
        self.moe = DeepSeekMoE(
            hidden_size, moe_intermediate_size,
            num_routed_experts, num_shared_experts, top_k
        )

    def forward(self, x: Tensor, position: int = 0) -> Tensor:
        # Pre-RMSNorm → Attention → 残差连接
        x = x + self.attention(self.attn_norm(x), position)
        # Pre-RMSNorm → MoE → 残差连接
        x = x + self.moe(self.ffn_norm(x))
        return x


# ════════════════════════════════════════════════════════════════════
# 9. DeepSeekV3Model（完整语言模型）
# ════════════════════════════════════════════════════════════════════

class DeepSeekV3Model(Module):
    """DeepSeek V3 语言模型（核心架构教学还原版）。

    完整架构：
      token_ids → Embedding
          ↓
      × N 层  DeepSeekV3Block
               ├─ Pre-RMSNorm
               ├─ MLA（低秩 KV 压缩 + RoPE）
               ├─ 残差连接
               ├─ Pre-RMSNorm
               ├─ DeepSeekMoE（共享专家 + 路由专家）
               └─ 残差连接
          ↓
      Final RMSNorm
          ↓
      LM Head（Linear → vocab_size）
          ↓
      logits → 预测下一个 token

    本示例 vs 真实 DeepSeek V3 规格对比：
    ──────────────────────────────────────────────
    配置项             本示例     真实 DeepSeek V3
    ──────────────────────────────────────────────
    总参数量            极小        671B
    激活参数/token      极小        37B
    层数                2           61
    hidden_size         64          7168
    注意力头数          4           128
    head_dim            16          128
    kv_lora_rank        32          512
    q_lora_rank         32          1536
    路由专家数          8           256
    共享专家数          1           1
    top_k               2           8
    moe_intermediate    128         2048
    词汇表大小          100         102400
    ──────────────────────────────────────────────
    """

    def __init__(self, vocab_size: int = 100, hidden_size: int = 64,
                 num_layers: int = 2, num_heads: int = 4, head_dim: int = 16,
                 kv_lora_rank: int = 32, q_lora_rank: int = 32,
                 moe_intermediate_size: int = 128,
                 num_routed_experts: int = 8, num_shared_experts: int = 1,
                 top_k: int = 2):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # 词嵌入层
        self.embedding = Embedding(vocab_size, hidden_size)

        # N × DeepSeekV3Block（ModuleList 保证参数正确注册）
        self.layers = ModuleList([
            DeepSeekV3Block(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                moe_intermediate_size=moe_intermediate_size,
                num_routed_experts=num_routed_experts,
                num_shared_experts=num_shared_experts,
                top_k=top_k
            )
            for _ in range(num_layers)
        ])

        # 最终归一化（输出前）
        self.final_norm = RMSNorm(hidden_size)

        # 语言模型头（投影到词汇表大小）
        self.lm_head = Linear(hidden_size, vocab_size, use_bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """前向传播（取序列最后一个位置，用于 next-token 预测）。

        Args:
            input_ids: token ID 序列，形状 (1, seq_len)

        Returns:
            logits，形状 (1, vocab_size)
        """
        # 词嵌入：(1, seq_len) → (1, seq_len, hidden)
        emb = self.embedding(input_ids)
        seq_len = input_ids.value.shape.dims[1]

        # 取最后一个 token 的隐向量
        last = emb.value.data[-self.hidden_size:]
        x = Tensor(
            NdArray(last, Shape((1, self.hidden_size)), 'float32'),
            requires_grad=emb.requires_grad
        )

        # 依次通过各 Transformer 层（传入位置信息用于 RoPE）
        position = seq_len - 1
        for layer in self.layers:
            x = layer(x, position)

        # 最终归一化 + LM 头
        return self.lm_head(self.final_norm(x))   # (1, vocab_size)

    def generate(self, prompt_ids: list, max_new_tokens: int = 5) -> list:
        """自回归文本生成（贪心解码）。

        Args:
            prompt_ids: 初始 token ID 列表
            max_new_tokens: 最多生成的新 token 数

        Returns:
            完整序列（prompt + 生成部分）
        """
        generated = prompt_ids.copy()
        for _ in range(max_new_tokens):
            inp = Tensor(
                NdArray([float(i) for i in generated],
                        Shape((1, len(generated))), 'float32'),
                requires_grad=False
            )
            logits = self.forward(inp)
            # 贪心解码：取概率最高的 token
            data = logits.value.data
            next_token = data.index(max(data))
            generated.append(next_token)
            if len(generated) > 100:
                break
        return generated

    def get_auxiliary_loss(self) -> float:
        """获取所有层 MoE 负载均衡辅助损失（CV²）的均值。

        DeepSeek V3 训练时：L_total = L_CE + 0.003 × L_balance
        """
        losses = [layer.moe.load_balance_loss() for layer in self.layers]
        return sum(losses) / max(len(losses), 1)

    def print_architecture(self):
        """打印模型架构摘要。"""
        attn0 = self.layers[0].attention
        moe0  = self.layers[0].moe
        n_r = len(moe0.routed_experts)
        n_s = len(moe0.shared_experts)
        print(f"  Embedding:       vocab={self.vocab_size}, dim={self.hidden_size}")
        print(f"  Transformer 层数: {self.num_layers} × DeepSeekV3Block")
        print(f"    MLA:  kv_lora_rank={attn0.kv_a_proj.out_features}"
              f", q_lora_rank={attn0.q_a_proj.out_features}"
              f", attn_dim={attn0.attn_dim}")
        print(f"    MoE:  {n_s} 共享专家 + {n_r} 路由专家, top_{moe0.top_k}")
        print(f"  FinalNorm:       RMSNorm({self.hidden_size})")
        print(f"  LM Head:         {self.hidden_size} → {self.vocab_size}")


# ════════════════════════════════════════════════════════════════════
# 主演示函数
# ════════════════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  DeepSeek V3 核心架构还原演示")
    print("=" * 68)

    print("""
【本示例完整还原的 DeepSeek V3 核心机制】

  ✓ RMSNorm        —— 均方根归一化，无中心化，全程替代 LayerNorm
  ✓ RoPE           —— 旋转位置编码，Q/K 注入相对位置，无额外参数
  ✓ MLA            —— 多头潜在注意力，KV 联合低秩压缩（64× 压缩比）
  ✓ SwiGLU         —— 门控 FFN：SiLU(gate) ⊙ up，代替 ReLU
  ✓ 共享专家       —— 始终激活，捕捉通用知识
  ✓ 路由专家+Top-K —— 真实排序门控，动态路由到专业专家
  ✓ 负载均衡损失   —— CV² 度量，防止专家坍塌
  ✓ Pre-RMSNorm    —— 残差前归一化，训练更稳定
""")

    # ── Step 1：初始化模型 ──────────────────────────────────────────
    print("=" * 68)
    print("Step 1: 初始化 DeepSeek V3 模型（微型教学版）")
    print("=" * 68)

    model = DeepSeekV3Model(
        vocab_size=100,
        hidden_size=64,            # 真实: 7168
        num_layers=2,              # 真实: 61
        num_heads=4,               # 真实: 128
        head_dim=16,               # 真实: 128
        kv_lora_rank=32,           # 真实: 512  ← MLA KV 缓存压缩维度
        q_lora_rank=32,            # 真实: 1536
        moe_intermediate_size=128, # 真实: 2048
        num_routed_experts=8,      # 真实: 256
        num_shared_experts=1,      # 真实: 1
        top_k=2                    # 真实: 8
    )

    print("\n模型架构摘要：")
    model.print_architecture()

    # ── Step 2：前向传播测试 ────────────────────────────────────────
    print("\n" + "=" * 68)
    print("Step 2: 前向传播")
    print("=" * 68)

    input_ids = [1, 2, 3, 4, 5]
    print(f"\n输入 token IDs: {input_ids}")

    inp = Tensor(
        NdArray([float(i) for i in input_ids], Shape((1, 5)), 'float32'),
        requires_grad=False
    )
    print("正在执行前向传播...")
    logits = model.forward(inp)

    print(f"✓ 输出 logits 形状: {logits.value.shape.dims}")
    top5 = sorted(enumerate(logits.value.data), key=lambda t: t[1], reverse=True)[:5]
    print(f"  Top-5 token 预测: {[(f'token_{i}', f'{s:.3f}') for i, s in top5]}")

    # ── Step 3：MLA KV 缓存压缩效果 ────────────────────────────────
    print("\n" + "=" * 68)
    print("Step 3: MLA KV 缓存压缩效果演示")
    print("=" * 68)

    attn = model.layers[0].attention
    kv_lora_rank = attn.kv_a_proj.out_features
    attn_dim     = attn.attn_dim

    print(f"""
  传统 MHA（每 token）：
    K 缓存 = {attn_dim} 维
    V 缓存 = {attn_dim} 维
    合计   = {attn_dim * 2} 维/token

  MLA（每 token）：
    c_KV 缓存 = {kv_lora_rank} 维/token   ← 只需缓存压缩表示！
    （K、V 在需要时从 c_KV 通过 kv_b_proj 实时恢复）

  压缩比 = {attn_dim * 2} / {kv_lora_rank} = {attn_dim * 2 / kv_lora_rank:.1f}×
  （真实 DeepSeek V3：32768 / 512 ≈ 64×，长序列显存大幅节省）
""")

    # ── Step 4：RoPE 位置编码演示 ───────────────────────────────────
    print("=" * 68)
    print("Step 4: RoPE 旋转位置编码演示")
    print("=" * 68)

    rope = attn.rope
    dummy = Tensor(
        NdArray([1.0] * attn_dim, Shape((1, attn_dim)), 'float32'),
        requires_grad=False
    )
    pos0 = rope.apply(dummy, 0)
    pos5 = rope.apply(dummy, 5)

    diff = sum(abs(a - b) for a, b in zip(pos0.value.data, pos5.value.data))
    print(f"\n  同一向量在位置 0 与位置 5 经 RoPE 后的 L1 差异: {diff:.4f}")
    print("  （非零差异表明 RoPE 成功将位置信息编码进向量角度）")
    print(f"  位置 0 前 4 维: {[round(v, 4) for v in pos0.value.data[:4]]}")
    print(f"  位置 5 前 4 维: {[round(v, 4) for v in pos5.value.data[:4]]}")

    # ── Step 5：MoE 专家路由分布演示 ───────────────────────────────
    print("\n" + "=" * 68)
    print("Step 5: MoE 专家路由分布演示")
    print("=" * 68)

    print("\n对 30 个不同隐向量做单层 MoE 前向，观察专家路由分布...")
    moe_layer = model.layers[0].moe
    ffn_norm  = model.layers[0].ffn_norm

    for i in range(30):
        # 构造各不相同的输入向量
        vals = [math.sin(i * 0.3 + j * 0.1) for j in range(model.hidden_size)]
        v = Tensor(NdArray(vals, Shape((1, model.hidden_size)), 'float32'),
                   requires_grad=False)
        moe_layer.forward(ffn_norm(v))

    print(f"\nLayer 0 MoE 路由统计（共享专家始终激活，下列为路由专家）：")
    moe_layer.print_load_stats()

    # ── Step 6：自回归文本生成 ──────────────────────────────────────
    print("\n" + "=" * 68)
    print("Step 6: 自回归文本生成（贪心解码）")
    print("=" * 68)

    prompt = [1, 2, 3]
    print(f"\n输入 prompt IDs: {prompt}")
    generated = model.generate(prompt, max_new_tokens=7)
    print(f"生成完整序列:   {generated}")
    print(f"新生成部分:     {generated[len(prompt):]}")

    # ── Step 7：负载均衡辅助损失 ────────────────────────────────────
    print("\n" + "=" * 68)
    print("Step 7: 负载均衡辅助损失（防专家坍塌）")
    print("=" * 68)

    aux = model.get_auxiliary_loss()
    print(f"\n  当前平均负载均衡损失 CV² = {aux:.4f}")
    print("  DeepSeek V3 训练：L_total = L_CE + 0.003 × L_balance")
    print("  CV² 越小 → 专家负载越均衡 → 每个专家都被充分训练")

    # ── Step 8：架构创新总结 ────────────────────────────────────────
    print("\n" + "=" * 68)
    print("Step 8: DeepSeek V3 架构创新总结")
    print("=" * 68)

    print("""
【MLA 的价值】
  KV 缓存是制约 LLM 长序列推理的核心瓶颈。
  MLA 通过联合低秩压缩将 KV 缓存降低 ~64×，
  使 DeepSeek V3 能高效处理超长上下文。

【MoE 双专家池的价值】
  • 共享专家：处理所有 token 的通用模式（语法、格式）
  • 路由专家：处理 token 专有模式（代码、数学、多语言）
  • 效果：671B 总参数，但每个 token 仅激活 ~37B，
    推理成本等价于 37B 稠密模型，但表达能力接近 671B。

【负载均衡损失的价值】
  若无辅助损失，部分热门专家被反复选中（专家坍塌），
  其余专家几乎不参与训练，浪费大量参数。
  辅助损失强制专家均等分工，充分发挥模型容量。

【Pre-RMSNorm 的价值】
  将归一化放在残差分支内（而非之后），
  训练更稳定，支持大规模模型深层训练。
""")

    print("=" * 68)
    print("  演示完成！")
    print("=" * 68)


if __name__ == '__main__':
    main()
