// SPDX-License-Identifier: AGPL-3.0-only
//
// Prefill-path Gated Delta Rule: run `num_tokens` sequential state
// updates in ONE dispatch per value head, holding the head's state
// slice in REGISTERS across the whole token loop. The decode kernel
// (`gated_delta_rule_decode`) reads + writes the full FP32 state from
// device memory every token — ~300 MB per token across Bonsai's 48 GDN
// layers, which is most of the per-token prefill cost once the
// projections are batched. Here the state streams once per chunk.
//
// Math per token is IDENTICAL to the decode kernel (same clamp, same
// HF update order — see that file's header):
//
//   hk_dot[c] = sum_j H[j, c] * k_t[j]
//   v_new[c]  = (v_t[c] - g_t * hk_dot[c]) * beta_t
//   H[j, c]   = g_t * H[j, c] + k_t[j] * v_new[c]
//   y_t[c]    = (sum_j H[j, c] * q_t[j]) / sqrt(k_dim)
//   ‖H‖_F > 1000 → scale down                (per token, as decode)
//
// One threadgroup of 128 threads per value head; thread `c` owns state
// column H[:, c] (128 floats in registers — DIM is a compile-time
// constant so the unrolled loops stay register-resident). Occupancy is
// register-limited, but the grid is only `num_v_heads` threadgroups
// and the arithmetic is register-local; the alternative (state in
// device memory) loses 100× on bandwidth.
//
// PRECONDITIONS (host-asserted): k_dim == v_dim == 128, threadgroup
// size 128.
//
// Layout (batch = 1):
//   h_state : float  [num_v_heads, 128, 128]        (in/out)
//   qkv     : bfloat [num_tokens, qkv_stride] — per token, Q at
//             [kh*128], K at [(num_k_heads + kh)*128], V at
//             [2*num_k_heads*128 + vh*128] (the conv1d-smoothed rows)
//   gate    : float  [num_tokens, num_v_heads]
//   beta    : float  [num_tokens, num_v_heads]
//   output  : bfloat [num_tokens, num_v_heads, 128]

#include <metal_stdlib>
using namespace metal;

constant float SSM_STATE_MAX_NORM_P = 1000.0f;
constant uint  DIM = 128u; // k_dim == v_dim

kernel void gated_delta_rule_prefill(
    device float        *h_state [[buffer(0)]],
    device const bfloat *qkv     [[buffer(1)]],
    device const float  *gate    [[buffer(2)]],
    device const float  *beta    [[buffer(3)]],
    device bfloat       *output  [[buffer(4)]],
    constant uint &num_tokens    [[buffer(5)]],
    constant uint &num_k_heads   [[buffer(6)]],
    constant uint &num_v_heads   [[buffer(7)]],
    constant uint &qkv_stride    [[buffer(8)]],
    uint  tg_idx    [[threadgroup_position_in_grid]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_grp  [[simdgroup_index_in_threadgroup]])
{
    const uint vh = tg_idx;
    if (vh >= num_v_heads) {
        return;
    }
    const uint head_repeat = num_v_heads / num_k_heads;
    const uint kh = vh / head_repeat;

    const uint q_off = kh * DIM;
    const uint k_off = (num_k_heads + kh) * DIM;
    const uint v_off = 2u * num_k_heads * DIM + vh * DIM;

    // This thread's state column H[:, tid], register-resident.
    device float *H = h_state + (ulong)vh * DIM * DIM;
    float hcol[DIM];
    #pragma unroll
    for (uint j = 0; j < DIM; ++j) {
        hcol[j] = H[j * DIM + tid];
    }

    threadgroup float smem_k[DIM];
    threadgroup float smem_q[DIM];
    threadgroup float norm_sums[4];
    threadgroup float head_norm_sq_storage;

    const float inv_sqrt_d = rsqrt(float(DIM));

    for (uint t = 0; t < num_tokens; ++t) {
        device const bfloat *row = qkv + (ulong)t * qkv_stride;
        smem_k[tid] = float(row[k_off + tid]);
        smem_q[tid] = float(row[q_off + tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float g_raw = gate[t * num_v_heads + vh];
        const float g  = fmin(fmax(g_raw, 1e-6f), 1.0f - 1e-6f);
        const float bt = beta[t * num_v_heads + vh];
        const float v_i = float(row[v_off + tid]);

        // hk_dot for this column, all register-local.
        float hk_dot = 0.0f;
        #pragma unroll
        for (uint j = 0; j < DIM; ++j) {
            hk_dot += hcol[j] * smem_k[j];
        }
        const float v_new_i = (v_i - g * hk_dot) * bt;

        // State update + output dot + norm accumulation in one pass.
        float q_dot = 0.0f;
        float local_sq = 0.0f;
        #pragma unroll
        for (uint j = 0; j < DIM; ++j) {
            const float h = g * hcol[j] + smem_k[j] * v_new_i;
            hcol[j] = h;
            q_dot += h * smem_q[j];
            local_sq += h * h;
        }

        // Per-token Frobenius clamp — bit-matches the decode kernel.
        float warp_sum = simd_sum(local_sq);
        if (simd_lane == 0) {
            norm_sums[simd_grp] = warp_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_grp == 0) {
            float s = (tid < 4u) ? norm_sums[tid] : 0.0f;
            s = simd_sum(s);
            if (tid == 0) {
                head_norm_sq_storage = s;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float head_norm_sq = head_norm_sq_storage;
        if (head_norm_sq > SSM_STATE_MAX_NORM_P * SSM_STATE_MAX_NORM_P) {
            const float scale = SSM_STATE_MAX_NORM_P * rsqrt(head_norm_sq);
            #pragma unroll
            for (uint j = 0; j < DIM; ++j) {
                hcol[j] *= scale;
            }
        }

        output[((ulong)t * num_v_heads + vh) * DIM + tid] =
            bfloat(q_dot * inv_sqrt_d);
        // smem_k/q are rewritten at the top of the next iteration.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint j = 0; j < DIM; ++j) {
        H[j * DIM + tid] = hcol[j];
    }
}
