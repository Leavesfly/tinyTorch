"""Neural-network oriented autograd operations.

These primitives support higher-level ``tinytorch.nn`` layers without forcing
layer modules to host their own autograd kernels.
"""

import math
from typing import List

from tinytorch.autograd.function import Function
from tinytorch.ndarr import NdArray, Shape
from tinytorch.utils import random as tt_random


class EmbeddingLookup(Function):
    """Embedding table lookup with weight gradients."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

    def forward(self, indices: NdArray, weight: NdArray) -> NdArray:
        self.save_for_backward(indices, weight)
        input_shape = indices.shape.dims
        output_shape = Shape(input_shape + (self.embedding_dim,))

        self._flat_indices = []
        output_data = []
        for idx_val in indices.data:
            idx = int(idx_val)
            if not 0 <= idx < self.num_embeddings:
                raise IndexError(f"Index {idx} out of range [0, {self.num_embeddings})")
            self._flat_indices.append(idx)
            start = idx * self.embedding_dim
            end = start + self.embedding_dim
            output_data.extend(weight.data[start:end])

        return NdArray(output_data, output_shape, weight.dtype)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        indices, weight = self.get_saved_tensors()

        grad_indices = NdArray.zeros(indices.shape, grad_output.dtype)
        grad_weight = NdArray.zeros(weight.shape, grad_output.dtype)

        for pos, idx in enumerate(self._flat_indices):
            if self.padding_idx is not None and idx == self.padding_idx:
                continue
            out_base = pos * self.embedding_dim
            w_base = idx * self.embedding_dim
            for d in range(self.embedding_dim):
                grad_weight.data[w_base + d] += grad_output.data[out_base + d]

        return [grad_indices, grad_weight]


class TimeSlice(Function):
    """Slice a sequence tensor along the time dimension."""

    def __init__(self, t: int, seq_len: int, input_size: int):
        super().__init__()
        self.t = t
        self.seq_len = seq_len
        self.input_size = input_size

    def forward(self, x: NdArray) -> NdArray:
        batch_size, seq_len, input_size = x.shape.dims
        if seq_len != self.seq_len or input_size != self.input_size:
            raise ValueError("Unexpected input shape for time slice")

        x_t_data = []
        for b in range(batch_size):
            start = b * seq_len * input_size + self.t * input_size
            end = start + input_size
            x_t_data.extend(x.data[start:end])
        return NdArray(x_t_data, Shape((batch_size, input_size)), x.dtype)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        grad_x = NdArray.zeros((grad_output.shape.dims[0], self.seq_len, self.input_size), grad_output.dtype)
        batch_size = grad_output.shape.dims[0]
        for b in range(batch_size):
            src_start = b * self.input_size
            src_end = src_start + self.input_size
            dst_start = b * self.seq_len * self.input_size + self.t * self.input_size
            dst_end = dst_start + self.input_size
            grad_x.data[dst_start:dst_end] = grad_output.data[src_start:src_end]
        return [grad_x]


class StackTime(Function):
    """Stack ``(B, H)`` tensors into ``(B, T, H)``."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, *steps: NdArray) -> NdArray:
        if not steps:
            raise ValueError("steps must not be empty")
        batch_size = steps[0].shape.dims[0]
        seq_len = len(steps)
        for step in steps:
            if step.shape.dims != (batch_size, self.hidden_size):
                raise ValueError("all step tensors must have shape (batch_size, hidden_size)")

        out = []
        for b in range(batch_size):
            for t in range(seq_len):
                start = b * self.hidden_size
                end = start + self.hidden_size
                out.extend(steps[t].data[start:end])
        return NdArray(out, Shape((batch_size, seq_len, self.hidden_size)), steps[0].dtype)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        batch_size, seq_len, hidden_size = grad_output.shape.dims
        grads = []
        for t in range(seq_len):
            g = []
            for b in range(batch_size):
                start = b * seq_len * hidden_size + t * hidden_size
                end = start + hidden_size
                g.extend(grad_output.data[start:end])
            grads.append(NdArray(g, Shape((batch_size, hidden_size)), grad_output.dtype))
        return grads


class SplitHeads(Function):
    """Split ``(B, T, E)`` into ``(B, H, T, D)``."""

    def __init__(self, num_heads: int, head_dim: int, embed_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = embed_dim

    def forward(self, x: NdArray) -> NdArray:
        batch_size, seq_len, _ = x.shape.dims
        data = []
        for b in range(batch_size):
            for h in range(self.num_heads):
                for s in range(seq_len):
                    for d in range(self.head_dim):
                        orig_idx = b * seq_len * self.embed_dim + s * self.embed_dim + h * self.head_dim + d
                        data.append(x.data[orig_idx])
        return NdArray(data, Shape((batch_size, self.num_heads, seq_len, self.head_dim)), x.dtype)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        batch_size, _, seq_len, _ = grad_output.shape.dims
        grad_x = [0.0] * (batch_size * seq_len * self.embed_dim)
        for b in range(batch_size):
            for h in range(self.num_heads):
                for s in range(seq_len):
                    for d in range(self.head_dim):
                        src_idx = (
                            b * self.num_heads * seq_len * self.head_dim
                            + h * seq_len * self.head_dim
                            + s * self.head_dim
                            + d
                        )
                        dst_idx = b * seq_len * self.embed_dim + s * self.embed_dim + h * self.head_dim + d
                        grad_x[dst_idx] = grad_output.data[src_idx]
        return [NdArray(grad_x, Shape((batch_size, seq_len, self.embed_dim)), grad_output.dtype)]


class MergeHeads(Function):
    """Merge ``(B, H, T, D)`` back into ``(B, T, E)``."""

    def __init__(self, num_heads: int, head_dim: int, embed_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = embed_dim

    def forward(self, x: NdArray) -> NdArray:
        batch_size, _, seq_len, _ = x.shape.dims
        data = []
        for b in range(batch_size):
            for s in range(seq_len):
                for h in range(self.num_heads):
                    for d in range(self.head_dim):
                        idx = (
                            b * self.num_heads * seq_len * self.head_dim
                            + h * seq_len * self.head_dim
                            + s * self.head_dim
                            + d
                        )
                        data.append(x.data[idx])
        return NdArray(data, Shape((batch_size, seq_len, self.embed_dim)), x.dtype)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        batch_size, seq_len, _ = grad_output.shape.dims
        grad_x = [0.0] * (batch_size * self.num_heads * seq_len * self.head_dim)
        for b in range(batch_size):
            for s in range(seq_len):
                for h in range(self.num_heads):
                    for d in range(self.head_dim):
                        src_idx = b * seq_len * self.embed_dim + s * self.embed_dim + h * self.head_dim + d
                        dst_idx = (
                            b * self.num_heads * seq_len * self.head_dim
                            + h * seq_len * self.head_dim
                            + s * self.head_dim
                            + d
                        )
                        grad_x[dst_idx] = grad_output.data[src_idx]
        return [NdArray(grad_x, Shape((batch_size, self.num_heads, seq_len, self.head_dim)), grad_output.dtype)]


def _mask_keep(mask_arr: NdArray, b: int, h: int, i: int, j: int) -> bool:
    dims = mask_arr.shape.dims
    data = mask_arr.data
    if len(dims) == 2:
        return data[i * dims[1] + j] > 0
    if len(dims) == 3:
        return data[b * dims[1] * dims[2] + i * dims[2] + j] > 0
    if len(dims) == 4:
        h_idx = 0 if dims[1] == 1 else h
        idx = b * dims[1] * dims[2] * dims[3] + h_idx * dims[2] * dims[3] + i * dims[3] + j
        return data[idx] > 0
    raise ValueError(f"Unsupported mask dims: {dims}")


class ScaledDotProductAttention(Function):
    """Scaled dot-product attention over already split heads."""

    def __init__(self, num_heads: int, head_dim: int, dropout: float = 0.0, training: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.training = training

    def forward(self, q: NdArray, k: NdArray, v: NdArray, mask: NdArray = None) -> NdArray:
        self.save_for_backward(q, k, v, mask)
        self._has_mask = mask is not None
        self._mask_shape = mask.shape if mask is not None else None
        batch_size, num_heads, q_len, head_dim = q.shape.dims
        _, _, kv_len, k_head_dim = k.shape.dims
        _, _, v_len, v_head_dim = v.shape.dims
        if k_head_dim != head_dim or v_head_dim != head_dim:
            raise ValueError("Q, K, V must have the same head_dim")
        if v_len != kv_len:
            raise ValueError("K and V must have the same sequence length")
        scale = 1.0 / math.sqrt(head_dim)
        q_data = q.data
        k_data = k.data
        v_data = v.data

        attn_weights = [0.0] * (batch_size * num_heads * q_len * kv_len)
        dropout_keep = [1.0] * len(attn_weights)
        output = [0.0] * (batch_size * num_heads * q_len * head_dim)

        for b in range(batch_size):
            for h in range(num_heads):
                qh_base = b * num_heads * q_len * head_dim + h * q_len * head_dim
                kh_base = b * num_heads * kv_len * head_dim + h * kv_len * head_dim
                ah_base = b * num_heads * q_len * kv_len + h * q_len * kv_len
                for i in range(q_len):
                    row_scores = []
                    qi_base = qh_base + i * head_dim
                    for j in range(kv_len):
                        s = 0.0
                        kj_base = kh_base + j * head_dim
                        for d in range(head_dim):
                            s += q_data[qi_base + d] * k_data[kj_base + d]
                        s *= scale
                        if mask is not None and not _mask_keep(mask, b, h, i, j):
                            s = -1e9
                        row_scores.append(s)

                    max_val = max(row_scores)
                    exp_row = [math.exp(x - max_val) for x in row_scores]
                    denom = sum(exp_row)
                    for j in range(kv_len):
                        a = exp_row[j] / denom
                        attn_idx = ah_base + i * kv_len + j
                        if self.training and self.dropout > 0.0:
                            keep = 0.0 if tt_random.random() < self.dropout else 1.0 / (1.0 - self.dropout)
                            dropout_keep[attn_idx] = keep
                            a *= keep
                        attn_weights[attn_idx] = a

                for i in range(q_len):
                    out_i_base = qh_base + i * head_dim
                    for d in range(head_dim):
                        val = 0.0
                        for j in range(kv_len):
                            attn_idx = ah_base + i * kv_len + j
                            v_idx = kh_base + j * head_dim + d
                            val += attn_weights[attn_idx] * v_data[v_idx]
                        output[out_i_base + d] = val

        self._attn_weights = attn_weights
        self._dropout_keep = dropout_keep
        return NdArray(output, Shape((batch_size, num_heads, q_len, head_dim)), q.dtype)

    def backward(self, grad_output: NdArray) -> List[NdArray]:
        q, k, v = self.get_saved_tensors()[:3]
        batch_size, num_heads, q_len, head_dim = q.shape.dims
        _, _, kv_len, _ = k.shape.dims
        scale = 1.0 / math.sqrt(head_dim)
        q_data = q.data
        k_data = k.data
        v_data = v.data
        go_data = grad_output.data

        grad_q = [0.0] * q.shape.size
        grad_k = [0.0] * k.shape.size
        grad_v = [0.0] * v.shape.size

        for b in range(batch_size):
            for h in range(num_heads):
                qh_base = b * num_heads * q_len * head_dim + h * q_len * head_dim
                kh_base = b * num_heads * kv_len * head_dim + h * kv_len * head_dim
                ah_base = b * num_heads * q_len * kv_len + h * q_len * kv_len
                grad_a = [[0.0 for _ in range(kv_len)] for _ in range(q_len)]
                for i in range(q_len):
                    go_i_base = qh_base + i * head_dim
                    for j in range(kv_len):
                        a_idx = ah_base + i * kv_len + j
                        a = self._attn_weights[a_idx]
                        vj_base = kh_base + j * head_dim
                        for d in range(head_dim):
                            grad_v[vj_base + d] += a * go_data[go_i_base + d]
                            grad_a[i][j] += go_data[go_i_base + d] * v_data[vj_base + d]

                if self.training and self.dropout > 0.0:
                    for i in range(q_len):
                        for j in range(kv_len):
                            keep_idx = ah_base + i * kv_len + j
                            grad_a[i][j] *= self._dropout_keep[keep_idx]

                grad_scores = [[0.0 for _ in range(kv_len)] for _ in range(q_len)]
                for i in range(q_len):
                    dot = 0.0
                    for j in range(kv_len):
                        a_idx = ah_base + i * kv_len + j
                        dot += grad_a[i][j] * self._attn_weights[a_idx]
                    for j in range(kv_len):
                        a_idx = ah_base + i * kv_len + j
                        a = self._attn_weights[a_idx]
                        grad_scores[i][j] = a * (grad_a[i][j] - dot)

                for i in range(q_len):
                    qi_base = qh_base + i * head_dim
                    for j in range(kv_len):
                        gs = grad_scores[i][j] * scale
                        kj_base = kh_base + j * head_dim
                        for d in range(head_dim):
                            grad_q[qi_base + d] += gs * k_data[kj_base + d]
                            grad_k[kj_base + d] += gs * q_data[qi_base + d]

        grads = [
            NdArray(grad_q, q.shape, grad_output.dtype),
            NdArray(grad_k, k.shape, grad_output.dtype),
            NdArray(grad_v, v.shape, grad_output.dtype),
        ]
        if self._has_mask:
            grads.append(NdArray.zeros(self._mask_shape, grad_output.dtype))
        return grads
