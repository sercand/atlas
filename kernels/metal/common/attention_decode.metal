// SPDX-License-Identifier: AGPL-3.0-only
//
// Decode-path scaled-dot-product attention against a contiguous KV
// cache. One threadgroup per head; threads inside the group cooperate
// on the K-dot-product, the softmax, and the V-weighted sum.
//
//   scores[s] = (Q[h] · K[s, h_kv]) / sqrt(head_dim)
//   softmax over s
//   out[h, d] = sum_s(softmax_s * V[s, h_kv, d])
//
// Supports Grouped-Query Attention: `num_heads` queries map to
// `num_kv_heads` keys/values via integer division (`h / (num_heads /
// num_kv_heads)`).
//
// Layout:
//   q   : bfloat [num_heads,   head_dim]            (one token)
//   k   : bfloat [seq_len, num_kv_heads, head_dim]  (cache)
//   v   : bfloat [seq_len, num_kv_heads, head_dim]
//   out : bfloat [num_heads,   head_dim]
//
// `seq_len` is capped at MAX_SEQ_DECODE because the per-token score
// vector lives in threadgroup memory. Long-context decode goes
// through the paged variant (separate kernel, future PR).

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SEQ_DECODE = 4096;
constant uint MAX_HEAD_DIM_DECODE = 256;
constant uint MAX_TG_DECODE = 1024;

kernel void attention_decode(
    constant uint  &seq_len      [[buffer(0)]],
    constant uint  &num_heads    [[buffer(1)]],
    constant uint  &num_kv_heads [[buffer(2)]],
    constant uint  &head_dim     [[buffer(3)]],
    constant float &scale        [[buffer(4)]],
    device const bfloat *q       [[buffer(5)]],
    device const bfloat *k       [[buffer(6)]],
    device const bfloat *v       [[buffer(7)]],
    device bfloat       *out     [[buffer(8)]],
    uint h       [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float scores[MAX_SEQ_DECODE];
    threadgroup float q_row[MAX_HEAD_DIM_DECODE];
    threadgroup float partial[MAX_TG_DECODE];
    threadgroup float red[32];
    threadgroup float max_score;
    threadgroup float sum_exp;

    if (h >= num_heads) {
        return;
    }
    // The score vector lives in threadgroup memory: positions past the
    // cap would read/write out of bounds in stages 2-5, so clamp hard.
    // Long-context decode belongs to a future paged variant.
    uint seq = min(seq_len, MAX_SEQ_DECODE);
    uint group = num_heads / num_kv_heads;
    uint kv_h  = h / group;
    uint lane = tid & 31u;
    uint sg   = tid >> 5u;
    uint num_simds = (tg_size + 31u) / 32u;

    // Stage 0: stage this head's query row — stage 1 walks it once
    // per KEY and re-reading it from device is a ~6× tax there.
    for (uint d = tid; d < head_dim; d += tg_size) {
        q_row[d] = float(q[h * head_dim + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 1: scores[s] = (Q[h] · K[s, kv_h]) * scale, tracking a
    // per-thread running max. Vectorized k loads when head_dim % 4 == 0
    // (always, for the models this backend serves).
    float local_max = -INFINITY;
    if ((head_dim & 3u) == 0u) {
        for (uint s = tid; s < seq; s += tg_size) {
            device const bfloat4 *k4 =
                reinterpret_cast<device const bfloat4*>(k + (s * num_kv_heads + kv_h) * head_dim);
            float dot = 0.0f;
            for (uint d4 = 0; d4 < head_dim / 4u; ++d4) {
                const bfloat4 kv = k4[d4];
                dot += q_row[d4 * 4u]      * float(kv.x)
                     + q_row[d4 * 4u + 1u] * float(kv.y)
                     + q_row[d4 * 4u + 2u] * float(kv.z)
                     + q_row[d4 * 4u + 3u] * float(kv.w);
            }
            float sc = dot * scale;
            scores[s] = sc;
            local_max = max(local_max, sc);
        }
    } else {
        for (uint s = tid; s < seq; s += tg_size) {
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += q_row[d] * float(k[(s * num_kv_heads + kv_h) * head_dim + d]);
            }
            float sc = dot * scale;
            scores[s] = sc;
            local_max = max(local_max, sc);
        }
    }

    // Stage 2: parallel max reduction (simd, then cross-simd).
    float simd_max_v = simd_max(local_max);
    if (lane == 0) {
        red[sg] = simd_max_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float m = (tid < num_simds) ? red[tid] : -INFINITY;
        m = simd_max(m);
        if (tid == 0) {
            max_score = m;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 3+4: exp(score - max) with a parallel sum reduction.
    float local_sum = 0.0f;
    for (uint s = tid; s < seq; s += tg_size) {
        float e = exp(scores[s] - max_score);
        scores[s] = e;
        local_sum += e;
    }
    float simd_sum_v = simd_sum(local_sum);
    if (lane == 0) {
        red[sg] = simd_sum_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float sum = (tid < num_simds) ? red[tid] : 0.0f;
        sum = simd_sum(sum);
        if (tid == 0) {
            sum_exp = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 5: out[h, d] = sum_s(softmax_s * V[s, kv_h, d]). With
    // tg_size ≥ head_dim, stripe the key walk: thread (stripe, d)
    // covers keys s ≡ stripe (mod nstripes), then a cross-stripe add
    // finishes each d — the serial walk shrinks by tg_size/head_dim
    // instead of leaving those threads idle. Smaller blocks keep the
    // plain d-strided loop.
    float inv_sum = 1.0f / sum_exp;
    if (tg_size >= head_dim) {
        uint nstripes = tg_size / head_dim;
        uint stripe = tid / head_dim;
        uint d = tid % head_dim;
        float acc = 0.0f;
        if (stripe < nstripes) {
            for (uint s = stripe; s < seq; s += nstripes) {
                acc += scores[s] * float(v[(s * num_kv_heads + kv_h) * head_dim + d]);
            }
        }
        partial[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint dd = tid; dd < head_dim; dd += tg_size) {
            float total = 0.0f;
            for (uint st = 0; st < nstripes; ++st) {
                total += partial[st * head_dim + dd];
            }
            out[h * head_dim + dd] = bfloat(total * inv_sum);
        }
    } else {
        for (uint d = tid; d < head_dim; d += tg_size) {
            float acc = 0.0f;
            for (uint s = 0; s < seq; ++s) {
                acc += scores[s] * float(v[(s * num_kv_heads + kv_h) * head_dim + d]);
            }
            out[h * head_dim + d] = bfloat(acc * inv_sum);
        }
    }
}
