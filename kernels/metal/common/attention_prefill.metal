// SPDX-License-Identifier: AGPL-3.0-only
//
// Prefill-path scaled-dot-product attention with causal masking.
// Multi-token query attends to a contiguous K/V history in a single
// kernel launch (no KV-cache-page indirection — that's the paged
// variant).
//
//   scores[m, s] = (Q[m, h] · K[s, h_kv]) / sqrt(head_dim)
//   masked: scores[m, s] = -∞ if s > m   (causal)
//   softmax over s
//   out[m, h, d] = sum_s(softmax_s * V[s, h_kv, d])
//
// One threadgroup per (query token, head) pair. Per-row score vector
// in threadgroup memory (cap MAX_SEQ_PREFILL).
//
// Layout:
//   q   : bfloat [num_tokens, num_heads,   head_dim]
//   k   : bfloat [seq_len,    num_kv_heads, head_dim]
//   v   : bfloat [seq_len,    num_kv_heads, head_dim]
//   out : bfloat [num_tokens, num_heads,   head_dim]

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SEQ_PREFILL = 4096;
constant uint MAX_HEAD_DIM_PREFILL = 256;

kernel void attention_prefill(
    constant uint  &num_tokens   [[buffer(0)]],
    constant uint  &seq_len      [[buffer(1)]],
    constant uint  &num_heads    [[buffer(2)]],
    constant uint  &num_kv_heads [[buffer(3)]],
    constant uint  &head_dim     [[buffer(4)]],
    constant float &scale        [[buffer(5)]],
    device const bfloat *q       [[buffer(6)]],
    device const bfloat *k       [[buffer(7)]],
    device const bfloat *v       [[buffer(8)]],
    device bfloat       *out     [[buffer(9)]],
    uint  tg_idx  [[threadgroup_position_in_grid]],
    uint  tid     [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]])
{
    threadgroup float scores[MAX_SEQ_PREFILL];
    threadgroup float max_score;
    threadgroup float sum_exp;

    // Flat 1-D grid dispatch: caller sends `num_heads * num_tokens`
    // threadgroups; we decode (m, h) here. Using uint3 builtins for
    // a 2-D grid would also work but Metal forbids mixing scalar
    // and vector position attributes in one entry point.
    uint h = tg_idx % num_heads;
    uint m = tg_idx / num_heads;
    if (m >= num_tokens || h >= num_heads) {
        return;
    }
    uint group = num_heads / num_kv_heads;
    uint kv_h  = h / group;
    // Causal cutoff: this query can attend to keys at positions
    // [0, m] inclusive — assumes Q occupies positions [0, num_tokens)
    // and K/V cover the same range.
    uint cutoff = m + 1u;

    // Stage 1: scores. Mask everything past the causal cutoff to -∞
    // so the softmax exp drives them to 0.
    for (uint s = tid; s < seq_len && s < MAX_SEQ_PREFILL; s += tg_size) {
        if (s >= cutoff) {
            scores[s] = -INFINITY;
            continue;
        }
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            float qv = float(q[(m * num_heads + h) * head_dim + d]);
            float kvv = float(k[(s * num_kv_heads + kv_h) * head_dim + d]);
            dot += qv * kvv;
        }
        scores[s] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: max reduction.
    if (tid == 0) {
        float mx = -INFINITY;
        for (uint s = 0; s < seq_len; ++s) {
            if (scores[s] > mx) mx = scores[s];
        }
        max_score = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 3: exp(score - max).
    for (uint s = tid; s < seq_len; s += tg_size) {
        scores[s] = exp(scores[s] - max_score);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 4: sum.
    if (tid == 0) {
        float sum = 0.0f;
        for (uint s = 0; s < seq_len; ++s) {
            sum += scores[s];
        }
        sum_exp = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 5: out[m, h, d] = sum_s(softmax_s * V[s, kv_h, d]).
    float inv_sum = 1.0f / sum_exp;
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint s = 0; s < seq_len; ++s) {
            float vv = float(v[(s * num_kv_heads + kv_h) * head_dim + d]);
            acc += scores[s] * inv_sum * vv;
        }
        out[(m * num_heads + h) * head_dim + d] = bfloat(acc);
    }
}

// Chunked-prefill variant: the query block occupies ABSOLUTE positions
// `[pos_base, pos_base + num_tokens)` of a K/V history of `seq_len`
// (= pos_base + num_tokens) entries — query m may attend keys
// [0, pos_base + m]. `pos_base = 0` reproduces `attention_prefill`.
kernel void attention_prefill_offset(
    constant uint  &num_tokens   [[buffer(0)]],
    constant uint  &seq_len      [[buffer(1)]],
    constant uint  &pos_base     [[buffer(2)]],
    constant uint  &num_heads    [[buffer(3)]],
    constant uint  &num_kv_heads [[buffer(4)]],
    constant uint  &head_dim     [[buffer(5)]],
    constant float &scale        [[buffer(6)]],
    device const bfloat *q       [[buffer(7)]],
    device const bfloat *k       [[buffer(8)]],
    device const bfloat *v       [[buffer(9)]],
    device bfloat       *out     [[buffer(10)]],
    uint  tg_idx  [[threadgroup_position_in_grid]],
    uint  tid     [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_grp  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float scores[MAX_SEQ_PREFILL];
    threadgroup float red[32];
    threadgroup float max_score;
    threadgroup float sum_exp;

    // Flat 1-D grid dispatch: caller sends `num_heads * num_tokens`
    // threadgroups; we decode (m, h) here. Using uint3 builtins for
    // a 2-D grid would also work but Metal forbids mixing scalar
    // and vector position attributes in one entry point.
    uint h = tg_idx % num_heads;
    uint m = tg_idx / num_heads;
    if (m >= num_tokens || h >= num_heads) {
        return;
    }
    uint group = num_heads / num_kv_heads;
    uint kv_h  = h / group;
    // Causal cutoff: query m sits at absolute position pos_base + m
    // and may attend keys [0, pos_base + m] inclusive.
    uint cutoff = pos_base + m + 1u;
    uint num_simds = (tg_size + 31u) / 32u;

    // Stage 0: this (m, h) pair's query row into threadgroup memory —
    // stage 1 walks it once per KEY, and re-reading it from device
    // cost ~6× on this kernel (MAX_HEAD_DIM_PREFILL caps head_dim).
    threadgroup float q_row[MAX_HEAD_DIM_PREFILL];
    for (uint d = tid; d < head_dim; d += tg_size) {
        q_row[d] = float(q[(m * num_heads + h) * head_dim + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 1: scores over the visible keys (vectorized k loads),
    // tracking a running per-thread max (keys past the cutoff are
    // never written — the later stages only walk [0, cutoff)).
    float local_max = -INFINITY;
    for (uint s = tid; s < cutoff && s < MAX_SEQ_PREFILL; s += tg_size) {
        device const bfloat4 *k4 =
            reinterpret_cast<device const bfloat4*>(k + (s * num_kv_heads + kv_h) * head_dim);
        float dot = 0.0f;
        for (uint d4 = 0; d4 < head_dim / 4u; ++d4) {
            const bfloat4 kv4 = k4[d4];
            dot += q_row[d4 * 4u]      * float(kv4.x)
                 + q_row[d4 * 4u + 1u] * float(kv4.y)
                 + q_row[d4 * 4u + 2u] * float(kv4.z)
                 + q_row[d4 * 4u + 3u] * float(kv4.w);
        }
        float sc = dot * scale;
        scores[s] = sc;
        local_max = max(local_max, sc);
    }

    // Stage 2: parallel max reduction (simd, then cross-simd).
    float simd_max_v = simd_max(local_max);
    if (simd_lane == 0) {
        red[simd_grp] = simd_max_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_grp == 0) {
        float v0 = (tid < num_simds) ? red[tid] : -INFINITY;
        v0 = simd_max(v0);
        if (tid == 0) {
            max_score = v0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 3+4: exp(score - max) with a parallel sum reduction.
    float local_sum = 0.0f;
    for (uint s = tid; s < cutoff; s += tg_size) {
        float e = exp(scores[s] - max_score);
        scores[s] = e;
        local_sum += e;
    }
    float simd_sum_v = simd_sum(local_sum);
    if (simd_lane == 0) {
        red[simd_grp] = simd_sum_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_grp == 0) {
        float v0 = (tid < num_simds) ? red[tid] : 0.0f;
        v0 = simd_sum(v0);
        if (tid == 0) {
            sum_exp = v0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 5: out[m, h, d] = sum_s(softmax_s * V[s, kv_h, d]).
    float inv_sum = 1.0f / sum_exp;
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint s = 0; s < cutoff; ++s) {
            float vv = float(v[(s * num_kv_heads + kv_h) * head_dim + d]);
            acc += scores[s] * inv_sum * vv;
        }
        out[(m * num_heads + h) * head_dim + d] = bfloat(acc);
    }
}
