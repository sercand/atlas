// SPDX-License-Identifier: AGPL-3.0-only
//
// Non-causal full self-attention. Used by ViT-style vision towers and
// any encoder-only block where every query attends to every key.
// Identical structure to `attention_prefill` but without the
// `s > m → -∞` mask — every (m, h) attends to all `seq_len` keys.
//
// Layout:
//   q   : bfloat [num_tokens, num_heads,   head_dim]
//   k   : bfloat [seq_len,    num_kv_heads, head_dim]
//   v   : bfloat [seq_len,    num_kv_heads, head_dim]
//   out : bfloat [num_tokens, num_heads,   head_dim]
//
// One threadgroup per (token, head); flat 1-D grid (Metal forbids
// mixing scalar and uint3 position attributes in one kernel).

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SEQ_FULL = 4096;

// Parallel-reduction rework (same contract as `attention_full`): one
// 128-thread threadgroup per (token, head), q row staged in threadgroup
// memory, bfloat4 K loads, simd max/sum reductions instead of the
// tid==0 sweeps above. head_dim must be ≤ MAX_HEAD_DIM_FULL and a
// multiple of 4 (vision: 72). ~2 orders of magnitude faster at the
// ViT's P=3-4k patch counts, where the serial sweeps dominated.
constant uint MAX_HEAD_DIM_FULL = 128;

kernel void attention_full_v2(
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
    uint  tg_size [[threads_per_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_grp  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float scores[MAX_SEQ_FULL];
    threadgroup float q_row[MAX_HEAD_DIM_FULL];
    threadgroup float vacc[4 * MAX_HEAD_DIM_FULL];
    threadgroup float red[32];
    threadgroup float max_score;
    threadgroup float sum_exp;

    uint h = tg_idx % num_heads;
    uint m = tg_idx / num_heads;
    if (m >= num_tokens || h >= num_heads) {
        return;
    }
    uint group = num_heads / num_kv_heads;
    uint kv_h  = h / group;
    uint num_simds = (tg_size + 31u) / 32u;

    for (uint d = tid; d < head_dim; d += tg_size) {
        q_row[d] = float(q[(m * num_heads + h) * head_dim + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scores over every key (no causal mask), tracking a running max.
    // Lane-per-key with vector loads: the K reads gather across rows,
    // but the deep per-lane ILP beats a coalesced one-key-per-simd
    // shape whose per-key simd_sum serializes the loop (measured).
    float local_max = -INFINITY;
    for (uint s = tid; s < seq_len && s < MAX_SEQ_FULL; s += tg_size) {
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

    float local_sum = 0.0f;
    for (uint s = tid; s < seq_len; s += tg_size) {
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

    // V pass, s-PARALLEL: each simdgroup strides the keys (s = sg,
    // sg+4, …) with lanes covering d — every V row read is coalesced
    // (the serial per-d version walked 2.3 KB-strided columns and was
    // ~900 ms/call at P=3072; this shape is ~30×). Partial per-simd
    // accumulators reduce through threadgroup memory at the end.
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint dsteps = (head_dim + 31u) / 32u;
    for (uint s = simd_grp; s < seq_len; s += num_simds) {
        const float p = scores[s];
        device const bfloat *vrow = v + (s * num_kv_heads + kv_h) * head_dim;
        for (uint j = 0; j < dsteps; ++j) {
            const uint d = simd_lane + j * 32u;
            if (d < head_dim) {
                acc[j] += p * float(vrow[d]);
            }
        }
    }
    for (uint j = 0; j < dsteps; ++j) {
        const uint d = simd_lane + j * 32u;
        if (d < head_dim) {
            vacc[simd_grp * MAX_HEAD_DIM_FULL + d] = acc[j];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / sum_exp;
    for (uint d = tid; d < head_dim; d += tg_size) {
        float total = 0.0f;
        for (uint sg = 0; sg < num_simds; ++sg) {
            total += vacc[sg * MAX_HEAD_DIM_FULL + d];
        }
        out[(m * num_heads + h) * head_dim + d] = bfloat(total * inv_sum);
    }
}

kernel void attention_full(
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
    threadgroup float scores[MAX_SEQ_FULL];
    threadgroup float max_score;
    threadgroup float sum_exp;

    uint h = tg_idx % num_heads;
    uint m = tg_idx / num_heads;
    if (m >= num_tokens || h >= num_heads) {
        return;
    }
    uint group = num_heads / num_kv_heads;
    uint kv_h  = h / group;

    // Stage 1: scores. No causal mask — every query sees every key.
    for (uint s = tid; s < seq_len && s < MAX_SEQ_FULL; s += tg_size) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            float qv = float(q[(m * num_heads + h) * head_dim + d]);
            float kvv = float(k[(s * num_kv_heads + kv_h) * head_dim + d]);
            dot += qv * kvv;
        }
        scores[s] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: max for numerical-stable softmax.
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
