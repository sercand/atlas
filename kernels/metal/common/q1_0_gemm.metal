// SPDX-License-Identifier: AGPL-3.0-only
//
// Prefill-path GEMM over keep-packed PrismML Q1_0 (ggml id 41) weights:
//
//   y[t, n] = sum_k W_dequant[n, k] * x[t, k]        t = 0 .. T-1
//
// Same packed layout and winning memory shape as `q1_0_gemv` (WORD-
// STRIDED lanes × 4 rows per simdgroup — see that file), extended with
// a TOKEN TILE: each threadgroup keeps accumulators for
// `TOKENS_PER_TG = 8` tokens, so every packed weight word (the decode
// path's bandwidth term — 3.4 GB per token when walked one token at a
// time) is loaded ONCE per 8 tokens. Per-token weight traffic drops
// 8×; the loop goes ALU-bound instead, which is the right trade for
// prefill (the x rows all sit in the SLC at prefill sizes).
//
// Grid: flat 1-D, `ceil(N/16) * ceil(T/8)` threadgroups of 128
// (4 simdgroups × 4 rows). Row tile decodes as `tg_idx % row_tiles`,
// token tile as `tg_idx / row_tiles` (Metal forbids mixing scalar and
// vector position attributes in one entry point).
//
// Preconditions: K % 128 == 0 (as gemv). N and T take any value —
// tail rows clamp reads / guard writes, the token tile clamps to T.

#include <metal_stdlib>
using namespace metal;

constant uint SIMDGROUP_SIZE  = 32u;
constant uint Q1_GROUP        = 128u;
constant uint Q1_BLOCK_BYTES  = 18u;
constant uint WORDS_PER_BLOCK = 8u; // 8 x u16 sign words = 128 bits
constant uint ROWS_PER_SG     = 4u;
constant uint ROWS_PER_TG     = 16u; // 4 simdgroups x 4 rows
constant uint TOKENS_PER_TG   = 8u;

// Signed accumulate of 16 already-converted inputs against one sign word.
static inline float q1_dot16f(ushort bits, thread const float *xv)
{
    float s = 0.0f;
    for (uint i = 0; i < 16; ++i) {
        s += ((bits >> i) & 1u) ? xv[i] : -xv[i];
    }
    return s;
}

// Core accumulation: 4 rows × up to 8 tokens per simdgroup. The
// per-word row scales + sign bits are hoisted into registers, then the
// token loop re-reads only x — weight bytes stream once per tile.
static inline void q1_gemm_accum(device const uchar  *packed,
                                 device const bfloat *x, // [T, K]
                                 uint N, uint K,
                                 uint row0, uint t0, uint tt,
                                 uint lane,
                                 thread float (&acc)[ROWS_PER_SG][TOKENS_PER_TG])
{
    const uint blocks_per_row = K / Q1_GROUP;
    const ulong row_bytes = (ulong)blocks_per_row * Q1_BLOCK_BYTES;
    device const uchar *rb[ROWS_PER_SG];
    for (uint r = 0; r < ROWS_PER_SG; ++r) {
        rb[r] = packed + (ulong)min(row0 + r, N - 1u) * row_bytes;
    }

    const uint words_per_row = blocks_per_row * WORDS_PER_BLOCK;
    for (uint w = lane; w < words_per_row; w += SIMDGROUP_SIZE) {
        const uint boff = (w / WORDS_PER_BLOCK) * Q1_BLOCK_BYTES;
        const uint soff = boff + 2u + (w % WORDS_PER_BLOCK) * 2u;

        float  d[ROWS_PER_SG];
        ushort bits[ROWS_PER_SG];
        for (uint r = 0; r < ROWS_PER_SG; ++r) {
            d[r] = float(*reinterpret_cast<device const half*>(rb[r] + boff));
            bits[r] = *reinterpret_cast<device const ushort*>(rb[r] + soff);
        }

        for (uint t = 0; t < tt; ++t) {
            device const bfloat4 *x4 =
                reinterpret_cast<device const bfloat4*>(x + (ulong)(t0 + t) * K);
            const bfloat4 xa = x4[w * 4u];
            const bfloat4 xb = x4[w * 4u + 1u];
            const bfloat4 xc = x4[w * 4u + 2u];
            const bfloat4 xd = x4[w * 4u + 3u];
            const float xv[16] = {
                float(xa.x), float(xa.y), float(xa.z), float(xa.w),
                float(xb.x), float(xb.y), float(xb.z), float(xb.w),
                float(xc.x), float(xc.y), float(xc.z), float(xc.w),
                float(xd.x), float(xd.y), float(xd.z), float(xd.w),
            };
            for (uint r = 0; r < ROWS_PER_SG; ++r) {
                acc[r][t] += d[r] * q1_dot16f(bits[r], xv);
            }
        }
    }
}

kernel void q1_0_gemm(
    constant uint &N            [[buffer(0)]],
    constant uint &K            [[buffer(1)]],
    constant uint &T            [[buffer(2)]],
    device const uchar  *packed [[buffer(3)]],
    device const bfloat *x      [[buffer(4)]], // [T, K]
    device bfloat       *y      [[buffer(5)]], // [T, N]
    uint   tg_idx        [[threadgroup_position_in_grid]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    const uint row_tiles = (N + ROWS_PER_TG - 1u) / ROWS_PER_TG;
    const uint row0 = (tg_idx % row_tiles) * ROWS_PER_TG + simd_group_id * ROWS_PER_SG;
    const uint t0 = (tg_idx / row_tiles) * TOKENS_PER_TG;
    if (row0 >= N || t0 >= T) {
        return;
    }
    const uint tt = min(TOKENS_PER_TG, T - t0);

    float acc[ROWS_PER_SG][TOKENS_PER_TG] = {{0.0f}};
    q1_gemm_accum(packed, x, N, K, row0, t0, tt, simd_lane_id, acc);

    for (uint r = 0; r < ROWS_PER_SG; ++r) {
        for (uint t = 0; t < tt; ++t) {
            const float row_sum = simd_sum(acc[r][t]);
            if (simd_lane_id == 0 && row0 + r < N) {
                y[(ulong)(t0 + t) * N + row0 + r] = bfloat(row_sum);
            }
        }
    }
}

// GEMM + residual-stream addition (batched `q1_0_gemv_resid`):
//   y[t, n] = x_resid[t, n] + sum_k W[n, k] * x[t, k]
kernel void q1_0_gemm_resid(
    constant uint &N             [[buffer(0)]],
    constant uint &K             [[buffer(1)]],
    constant uint &T             [[buffer(2)]],
    device const uchar  *packed  [[buffer(3)]],
    device const bfloat *x       [[buffer(4)]], // [T, K]
    device const bfloat *x_resid [[buffer(5)]], // [T, N]
    device bfloat       *y       [[buffer(6)]], // [T, N]
    uint   tg_idx        [[threadgroup_position_in_grid]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    const uint row_tiles = (N + ROWS_PER_TG - 1u) / ROWS_PER_TG;
    const uint row0 = (tg_idx % row_tiles) * ROWS_PER_TG + simd_group_id * ROWS_PER_SG;
    const uint t0 = (tg_idx / row_tiles) * TOKENS_PER_TG;
    if (row0 >= N || t0 >= T) {
        return;
    }
    const uint tt = min(TOKENS_PER_TG, T - t0);

    float acc[ROWS_PER_SG][TOKENS_PER_TG] = {{0.0f}};
    q1_gemm_accum(packed, x, N, K, row0, t0, tt, simd_lane_id, acc);

    for (uint r = 0; r < ROWS_PER_SG; ++r) {
        for (uint t = 0; t < tt; ++t) {
            const float row_sum = simd_sum(acc[r][t]);
            if (simd_lane_id == 0 && row0 + r < N) {
                const ulong o = (ulong)(t0 + t) * N + row0 + r;
                y[o] = bfloat(row_sum + float(x_resid[o]));
            }
        }
    }
}
