// SPDX-License-Identifier: AGPL-3.0-only
//
// Fused Q1_0 GEMV + SwiGLU input activation (down_proj hot path):
//
//   y[n] = sum_k W_dequant[n, k] * (silu(gate[k]) * up[k])
//
// and the `_resid` variant that folds the residual stream addition:
//
//   y[n] = x_resid[n] + sum_k W[n, k] * (silu(gate[k]) * up[k])
//
// Weight layout is keep-packed PrismML Q1_0 exactly as in
// `q1_0_gemv.metal` (18-byte blocks: fp16 d + 128 LSB-first sign bits,
// 8 x u16 sign words per block); the input side is the SwiGLU pair
// instead of a plain x. Threadgroup layout = 4 rows × 32 lanes (one
// simdgroup per row), grid ceil(N/4), 128 threads — identical to the
// base kernel.

#include <metal_stdlib>
using namespace metal;

constant uint ROWS_PER_TG_SG    = 4u;
constant uint SIMDGROUP_SIZE_SG = 32u;
constant uint Q1SG_GROUP        = 128u;
constant uint Q1SG_BLOCK_BYTES  = 18u;
constant uint Q1SG_WORDS_PER_BLOCK = 8u; // 8 x u16 sign words = 128 bits

// Signed accumulate of 16 SwiGLU-activated inputs against one sign word.
static inline float q1_swiglu16(
    ushort bits,
    device const bfloat4 *gate4,
    device const bfloat4 *up4,
    uint w)
{
    float s = 0.0f;
    for (uint q = 0; q < 4; ++q) {
        const bfloat4 gv = gate4[w * 4u + q];
        const bfloat4 uv = up4[w * 4u + q];
        const float g[4] = {float(gv.x), float(gv.y), float(gv.z), float(gv.w)};
        const float u[4] = {float(uv.x), float(uv.y), float(uv.z), float(uv.w)};
        for (uint i = 0; i < 4; ++i) {
            // SwiGLU in FP32 — sigmoid stays stable for large |gate|.
            const float f = (g[i] / (1.0f + exp(-g[i]))) * u[i];
            s += ((bits >> (q * 4u + i)) & 1u) ? f : -f;
        }
    }
    return s;
}

kernel void q1_0_gemv_silu_gate(
    constant uint &N            [[buffer(0)]],
    constant uint &K            [[buffer(1)]],
    device const uchar  *packed [[buffer(2)]],
    device const bfloat *gate   [[buffer(3)]],
    device const bfloat *up     [[buffer(4)]],
    device bfloat       *y      [[buffer(5)]],
    uint   tg_idx        [[threadgroup_position_in_grid]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    const uint row = tg_idx * ROWS_PER_TG_SG + simd_group_id;
    if (row >= N) {
        return;
    }

    const uint blocks_per_row = K / Q1SG_GROUP;
    const uint words_per_row  = blocks_per_row * Q1SG_WORDS_PER_BLOCK;
    device const uchar *row_base =
        packed + (ulong)row * blocks_per_row * Q1SG_BLOCK_BYTES;
    device const bfloat4 *gate4 = reinterpret_cast<device const bfloat4*>(gate);
    device const bfloat4 *up4   = reinterpret_cast<device const bfloat4*>(up);

    float acc = 0.0f;
    for (uint w = simd_lane_id; w < words_per_row; w += SIMDGROUP_SIZE_SG) {
        const uint blk  = w / Q1SG_WORDS_PER_BLOCK;
        const uint wblk = w % Q1SG_WORDS_PER_BLOCK;
        device const uchar *bp = row_base + blk * Q1SG_BLOCK_BYTES;
        const float d = float(*reinterpret_cast<device const half*>(bp));
        const ushort bits =
            *reinterpret_cast<device const ushort*>(bp + 2u + wblk * 2u);

        acc += d * q1_swiglu16(bits, gate4, up4, w);
    }

    const float row_sum = simd_sum(acc);
    if (simd_lane_id == 0) {
        y[row] = bfloat(row_sum);
    }
}

kernel void q1_0_gemv_silu_gate_resid(
    constant uint &N             [[buffer(0)]],
    constant uint &K             [[buffer(1)]],
    device const uchar  *packed  [[buffer(2)]],
    device const bfloat *gate    [[buffer(3)]],
    device const bfloat *up      [[buffer(4)]],
    device const bfloat *x_resid [[buffer(5)]],
    device bfloat       *y       [[buffer(6)]],
    uint   tg_idx        [[threadgroup_position_in_grid]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    const uint row = tg_idx * ROWS_PER_TG_SG + simd_group_id;
    if (row >= N) {
        return;
    }

    const uint blocks_per_row = K / Q1SG_GROUP;
    const uint words_per_row  = blocks_per_row * Q1SG_WORDS_PER_BLOCK;
    device const uchar *row_base =
        packed + (ulong)row * blocks_per_row * Q1SG_BLOCK_BYTES;
    device const bfloat4 *gate4 = reinterpret_cast<device const bfloat4*>(gate);
    device const bfloat4 *up4   = reinterpret_cast<device const bfloat4*>(up);

    float acc = 0.0f;
    for (uint w = simd_lane_id; w < words_per_row; w += SIMDGROUP_SIZE_SG) {
        const uint blk  = w / Q1SG_WORDS_PER_BLOCK;
        const uint wblk = w % Q1SG_WORDS_PER_BLOCK;
        device const uchar *bp = row_base + blk * Q1SG_BLOCK_BYTES;
        const float d = float(*reinterpret_cast<device const half*>(bp));
        const ushort bits =
            *reinterpret_cast<device const ushort*>(bp + 2u + wblk * 2u);

        acc += d * q1_swiglu16(bits, gate4, up4, w);
    }

    const float row_sum = simd_sum(acc);
    if (simd_lane_id == 0) {
        y[row] = bfloat(row_sum + float(x_resid[row]));
    }
}
