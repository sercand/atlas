// SPDX-License-Identifier: AGPL-3.0-only
//
// Fused dual-output Q1_0 GEMV with a shared input vector:
//
//   gate_y[n] = sum_k GATE_dequant[n, k] * x[k]
//   up_y[n]   = sum_k UP_dequant[n, k]   * x[k]
//
// One x[] read drives both projections — halves the input-side
// bandwidth and removes a kernel launch on the FFN entry (gate_proj +
// up_proj always share (N, K) in a SwiGLU FFN). Weight layout is
// keep-packed PrismML Q1_0 as in `q1_0_gemv.metal`. Threadgroup
// layout = 4 rows × 32 lanes, grid ceil(N/4), 128 threads.

#include <metal_stdlib>
using namespace metal;

constant uint ROWS_PER_TG_GU    = 4u;
constant uint SIMDGROUP_SIZE_GU = 32u;
constant uint Q1GU_GROUP        = 128u;
constant uint Q1GU_BLOCK_BYTES  = 18u;
constant uint Q1GU_WORDS_PER_BLOCK = 8u; // 8 x u16 sign words = 128 bits

kernel void q1_0_gemv_gate_up(
    constant uint &N                 [[buffer(0)]],
    constant uint &K                 [[buffer(1)]],
    device const uchar  *gate_packed [[buffer(2)]],
    device const uchar  *up_packed   [[buffer(3)]],
    device const bfloat *x           [[buffer(4)]],
    device bfloat       *gate_y      [[buffer(5)]],
    device bfloat       *up_y        [[buffer(6)]],
    uint   tg_idx        [[threadgroup_position_in_grid]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    const uint row = tg_idx * ROWS_PER_TG_GU + simd_group_id;
    if (row >= N) {
        return;
    }

    const uint blocks_per_row = K / Q1GU_GROUP;
    const uint words_per_row  = blocks_per_row * Q1GU_WORDS_PER_BLOCK;
    const ulong row_off = (ulong)row * blocks_per_row * Q1GU_BLOCK_BYTES;
    device const uchar *grow = gate_packed + row_off;
    device const uchar *urow = up_packed + row_off;
    device const bfloat4 *x4 = reinterpret_cast<device const bfloat4*>(x);

    float gacc = 0.0f;
    float uacc = 0.0f;
    for (uint w = simd_lane_id; w < words_per_row; w += SIMDGROUP_SIZE_GU) {
        const uint blk  = w / Q1GU_WORDS_PER_BLOCK;
        const uint wblk = w % Q1GU_WORDS_PER_BLOCK;
        const uint boff = blk * Q1GU_BLOCK_BYTES;
        const float gd = float(*reinterpret_cast<device const half*>(grow + boff));
        const float ud = float(*reinterpret_cast<device const half*>(urow + boff));
        const ushort gbits =
            *reinterpret_cast<device const ushort*>(grow + boff + 2u + wblk * 2u);
        const ushort ubits =
            *reinterpret_cast<device const ushort*>(urow + boff + 2u + wblk * 2u);

        // 16 x values for this word: elements [w*16, w*16+16).
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
        float gs = 0.0f;
        float us = 0.0f;
        for (uint i = 0; i < 16; ++i) {
            gs += ((gbits >> i) & 1u) ? xv[i] : -xv[i];
            us += ((ubits >> i) & 1u) ? xv[i] : -xv[i];
        }
        gacc += gd * gs;
        uacc += ud * us;
    }

    const float g_sum = simd_sum(gacc);
    const float u_sum = simd_sum(uacc);
    if (simd_lane_id == 0) {
        gate_y[row] = bfloat(g_sum);
        up_y[row] = bfloat(u_sum);
    }
}
