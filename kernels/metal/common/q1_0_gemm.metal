// SPDX-License-Identifier: AGPL-3.0-only
//
// Prefill-path GEMM over keep-packed PrismML Q1_0 (ggml id 41) weights:
//
//   y[t, n] = sum_k W_dequant[n, k] * x[t, k]        t = 0 .. T-1
//
// Tiled simdgroup-matrix kernel, fourth shape iteration. What the
// earlier cuts taught (all measured on the 17408×5120 FFN projection
// at T = 256):
// - v1 scalar select/add with token accumulators: ~34 ms — ALU-bound
//   on the scalar pipe, same ops/MAC as the decode gemv.
// - v2/v3 simdgroup-matrix with the x tile staged bf16→half into
//   threadgroup memory: ~34 ms — the x staging re-ran once per ROW
//   tile (544× per call), and 5-6 tile loads per 4-8 MMAs kept the
//   matrix pipe waiting.
// - v4 (this): X arrives as DEVICE HALF ([T, K], pre-converted once
//   per input by the `bf16_to_half` helper — the conversion cost moves
//   out of the row-tile loop entirely). A fragments simdgroup_load
//   straight from device (the x rows live in the SLC at prefill
//   sizes); only the dequantized weight tile is staged (k-major, so B
//   loads are contiguous). 128-token tile → each simdgroup holds 16
//   fp32 accumulators: per 8-deep k-step it issues 4 A + 4 B loads
//   for 16 MMAs (0.5 loads/MMA vs v3's 0.75).
//
// Weight dequant: one Q1 block (128 k) per stage; d is fp16 natively
// so ±d in half is exact; the cost amortizes over the 128 tokens.
//
// Tails: token/row loads clamp; stores go through a per-simdgroup
// staging patch with bounds guards. CALLER CONTRACT: x must be padded
// so row reads up to ceil(T/8)*8 are in bounds (the prefill mats are
// tile-rounded; simdgroup_load cannot clamp row-by-row).
//
// Grid: flat `ceil(N/32) * ceil(T/128)` threadgroups of 128 threads.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint Q1_BLOCK_BYTES = 18u;
constant uint BM = 32u;  // weight rows per threadgroup tile
constant uint BN = 128u; // tokens per threadgroup tile (32 per simdgroup)
constant uint BK = 128u; // k-depth per stage = one Q1 block

// Dequantize the [BM, BK] weight tile at (row0, block kb) K-MAJOR into
// `wt` ([BK][BM], row stride BM). 128 threads; rows clamp at N-1.
static inline void stage_weights(device const uchar *packed,
                                 threadgroup half   *wt,
                                 uint N, uint K,
                                 uint row0, uint kb,
                                 uint tid)
{
    const ulong row_bytes = (ulong)(K / BK) * Q1_BLOCK_BYTES;
    for (uint task = tid; task < BM * 8u; task += 128u) {
        const uint r = task / 8u;
        const uint w = task % 8u;
        const uint row = min(row0 + r, N - 1u);
        device const uchar *bp = packed + (ulong)row * row_bytes + (ulong)kb * Q1_BLOCK_BYTES;
        const half d = *reinterpret_cast<device const half*>(bp);
        const ushort bits = *reinterpret_cast<device const ushort*>(bp + 2u + w * 2u);
        threadgroup half *dst = wt + (w * 16u) * BM + r;
        for (uint i = 0; i < 16u; ++i) {
            dst[i * BM] = ((bits >> i) & 1u) ? d : -d;
        }
    }
}

// Accumulate this simdgroup's 32-token stripe × 32 columns over all K.
// C[i][j] covers tokens [t0 + 32·sg + 8i, +8) × rows [row0 + 8j, +8).
static inline void gemm_tile(device const uchar *packed,
                             device const half  *x, // [T, K] padded
                             threadgroup half   *wt,
                             uint N, uint K,
                             uint row0, uint t0,
                             uint tid, uint sgid,
                             thread simdgroup_float8x8 (&C)[4][4])
{
    device const half *xs = x + (ulong)(t0 + sgid * 32u) * K;
    const uint blocks = K / BK;
    for (uint kb = 0; kb < blocks; ++kb) {
        stage_weights(packed, wt, N, K, row0, kb, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k8 = 0; k8 < BK / 8u; ++k8) {
            device const half *xk = xs + kb * BK + k8 * 8u;
            simdgroup_half8x8 a[4];
            simdgroup_load(a[0], xk, K);
            simdgroup_load(a[1], xk + 8u * K, K);
            simdgroup_load(a[2], xk + 16u * K, K);
            simdgroup_load(a[3], xk + 24u * K, K);
            for (uint j = 0; j < 4u; ++j) {
                simdgroup_half8x8 b;
                simdgroup_load(b, wt + (k8 * 8u) * BM + j * 8u, BM);
                simdgroup_multiply_accumulate(C[0][j], a[0], b, C[0][j]);
                simdgroup_multiply_accumulate(C[1][j], a[1], b, C[1][j]);
                simdgroup_multiply_accumulate(C[2][j], a[2], b, C[2][j]);
                simdgroup_multiply_accumulate(C[3][j], a[3], b, C[3][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Guarded store of one simdgroup's 16 C tiles through a per-simdgroup
// staging patch. `resid` is null for the plain variant.
static inline void store_tiles(threadgroup float   *sg_ct, // [64] per SG
                               device const bfloat *resid,
                               device bfloat       *y,
                               uint N, uint T,
                               uint row0, uint t0_sg,
                               uint lane,
                               thread simdgroup_float8x8 (&C)[4][4])
{
    for (uint i = 0; i < 4u; ++i) {
        for (uint j = 0; j < 4u; ++j) {
            simdgroup_store(C[i][j], sg_ct, 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64u; e += 32u) {
                const uint tr = t0_sg + i * 8u + e / 8u;
                const uint nc = row0 + j * 8u + e % 8u;
                if (tr < T && nc < N) {
                    const ulong o = (ulong)tr * N + nc;
                    const float r = resid ? float(resid[o]) : 0.0f;
                    y[o] = bfloat(sg_ct[e] + r);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void q1_0_gemm(
    constant uint &N            [[buffer(0)]],
    constant uint &K            [[buffer(1)]],
    constant uint &T            [[buffer(2)]],
    device const uchar  *packed [[buffer(3)]],
    device const half   *x      [[buffer(4)]], // [T, K] half, tile-padded
    device bfloat       *y      [[buffer(5)]], // [T, N]
    uint tg_idx [[threadgroup_position_in_grid]],
    uint tid    [[thread_position_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    threadgroup half wt[BK * BM];
    threadgroup float ct[4 * 64];

    const uint row_tiles = (N + BM - 1u) / BM;
    const uint row0 = (tg_idx % row_tiles) * BM;
    const uint t0 = (tg_idx / row_tiles) * BN;

    simdgroup_float8x8 C[4][4];
    for (uint i = 0; i < 4u; ++i) {
        for (uint j = 0; j < 4u; ++j) {
            C[i][j] = simdgroup_float8x8(0.0f);
        }
    }
    gemm_tile(packed, x, wt, N, K, row0, t0, tid, sgid, C);
    store_tiles(ct + sgid * 64u, nullptr, y, N, T, row0, t0 + sgid * 32u, lane, C);
}

// GEMM + residual-stream addition (batched `q1_0_gemv_resid`):
//   y[t, n] = x_resid[t, n] + sum_k W[n, k] * x[t, k]
kernel void q1_0_gemm_resid(
    constant uint &N             [[buffer(0)]],
    constant uint &K             [[buffer(1)]],
    constant uint &T             [[buffer(2)]],
    device const uchar  *packed  [[buffer(3)]],
    device const half   *x       [[buffer(4)]], // [T, K] half, tile-padded
    device const bfloat *x_resid [[buffer(5)]], // [T, N]
    device bfloat       *y       [[buffer(6)]], // [T, N]
    uint tg_idx [[threadgroup_position_in_grid]],
    uint tid    [[thread_position_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    threadgroup half wt[BK * BM];
    threadgroup float ct[4 * 64];

    const uint row_tiles = (N + BM - 1u) / BM;
    const uint row0 = (tg_idx % row_tiles) * BM;
    const uint t0 = (tg_idx / row_tiles) * BN;

    simdgroup_float8x8 C[4][4];
    for (uint i = 0; i < 4u; ++i) {
        for (uint j = 0; j < 4u; ++j) {
            C[i][j] = simdgroup_float8x8(0.0f);
        }
    }
    gemm_tile(packed, x, wt, N, K, row0, t0, tid, sgid, C);
    store_tiles(ct + sgid * 64u, x_resid, y, N, T, row0, t0 + sgid * 32u, lane, C);
}
