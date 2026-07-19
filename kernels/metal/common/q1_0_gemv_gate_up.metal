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
// keep-packed PrismML Q1_0 as in `q1_0_gemv.metal`, and so is the
// threadgroup shape: 4 simdgroups × 4 rows per simdgroup = 16 rows per
// 128-thread threadgroup, grid ceil(N/16), WORD-STRIDED lanes
// (adjacent lanes read adjacent sign words / x bytes — coalesced) with
// each x load + convert shared across all 8 weight streams
// (4 rows × gate/up).

#include <metal_stdlib>
using namespace metal;

constant uint SIMDGROUP_SIZE_GU = 32u;
constant uint Q1GU_GROUP        = 128u;
constant uint Q1GU_BLOCK_BYTES  = 18u;
constant uint Q1GU_WORDS_PER_BLOCK = 8u; // 8 x u16 sign words = 128 bits
constant uint ROWS_PER_SG_GU    = 4u;
constant uint ROWS_PER_TG_GU    = 16u;

static inline float q1gu_dot16f(ushort bits, thread const float *xv)
{
    float s = 0.0f;
    for (uint i = 0; i < 16; ++i) {
        s += ((bits >> i) & 1u) ? xv[i] : -xv[i];
    }
    return s;
}

// Word-strided 4-row × 2-stream accumulator (see q1_0_gemv.metal's
// q1_rows4_accum for the layout rationale).
static inline void q1gu_rows4_accum(device const uchar *gate_packed,
                                    device const uchar *up_packed,
                                    uint N,
                                    uint blocks_per_row,
                                    uint row0,
                                    device const bfloat4 *x4,
                                    uint lane,
                                    thread float gacc[4],
                                    thread float uacc[4])
{
    const ulong row_bytes = (ulong)blocks_per_row * Q1GU_BLOCK_BYTES;
    device const uchar *grb[4];
    device const uchar *urb[4];
    for (uint r = 0; r < ROWS_PER_SG_GU; ++r) {
        const ulong off = (ulong)min(row0 + r, N - 1u) * row_bytes;
        grb[r] = gate_packed + off;
        urb[r] = up_packed + off;
    }

    const uint words_per_row = blocks_per_row * Q1GU_WORDS_PER_BLOCK;
    for (uint w = lane; w < words_per_row; w += SIMDGROUP_SIZE_GU) {
        const uint boff = (w / Q1GU_WORDS_PER_BLOCK) * Q1GU_BLOCK_BYTES;
        const uint soff = boff + 2u + (w % Q1GU_WORDS_PER_BLOCK) * 2u;

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
        for (uint r = 0; r < ROWS_PER_SG_GU; ++r) {
            const float gd =
                float(*reinterpret_cast<device const half*>(grb[r] + boff));
            const ushort gbits =
                *reinterpret_cast<device const ushort*>(grb[r] + soff);
            gacc[r] += gd * q1gu_dot16f(gbits, xv);
            const float ud =
                float(*reinterpret_cast<device const half*>(urb[r] + boff));
            const ushort ubits =
                *reinterpret_cast<device const ushort*>(urb[r] + soff);
            uacc[r] += ud * q1gu_dot16f(ubits, xv);
        }
    }
}

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
    const uint row0 = tg_idx * ROWS_PER_TG_GU + simd_group_id * ROWS_PER_SG_GU;
    if (row0 >= N) {
        return;
    }
    device const bfloat4 *x4 = reinterpret_cast<device const bfloat4*>(x);
    float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float uacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    q1gu_rows4_accum(gate_packed, up_packed, N, K / Q1GU_GROUP, row0, x4,
                     simd_lane_id, gacc, uacc);

    for (uint r = 0; r < ROWS_PER_SG_GU; ++r) {
        const float g_sum = simd_sum(gacc[r]);
        const float u_sum = simd_sum(uacc[r]);
        if (simd_lane_id == 0 && row0 + r < N) {
            gate_y[row0 + r] = bfloat(g_sum);
            up_y[row0 + r] = bfloat(u_sum);
        }
    }
}

// Single-output SwiGLU variant: same dual weight streams, but the
// epilogue writes the ACTIVATED vector directly —
//   act[n] = silu(gate_y[n]) * up_y[n]
// so the FFN entry needs no separate elementwise silu dispatch and the
// down_proj can consume `act` with a plain gemv.
kernel void q1_0_gemv_gate_up_act(
    constant uint &N                 [[buffer(0)]],
    constant uint &K                 [[buffer(1)]],
    device const uchar  *gate_packed [[buffer(2)]],
    device const uchar  *up_packed   [[buffer(3)]],
    device const bfloat *x           [[buffer(4)]],
    device bfloat       *act         [[buffer(5)]],
    uint   tg_idx        [[threadgroup_position_in_grid]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    const uint row0 = tg_idx * ROWS_PER_TG_GU + simd_group_id * ROWS_PER_SG_GU;
    if (row0 >= N) {
        return;
    }
    device const bfloat4 *x4 = reinterpret_cast<device const bfloat4*>(x);
    float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float uacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    q1gu_rows4_accum(gate_packed, up_packed, N, K / Q1GU_GROUP, row0, x4,
                     simd_lane_id, gacc, uacc);

    for (uint r = 0; r < ROWS_PER_SG_GU; ++r) {
        const float g = simd_sum(gacc[r]);
        const float u = simd_sum(uacc[r]);
        if (simd_lane_id == 0 && row0 + r < N) {
            const float sig = 1.0f / (1.0f + exp(-g));
            act[row0 + r] = bfloat(g * sig * u);
        }
    }
}

// Row-planar variant (`PackedQ1Planar`, opt-in via ATLAS_Q1_PLANAR=1):
// per row, all 16-byte sign runs first (aligned uint4 per block) then
// all fp16 scales — see q1_0_gemv.metal for the layout.
kernel void q1_0_gemv_gate_up_planar(
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
    const uint row0 = tg_idx * ROWS_PER_TG_GU + simd_group_id * ROWS_PER_SG_GU;
    if (row0 >= N) {
        return;
    }

    const uint blocks_per_row = K / Q1GU_GROUP;
    const ulong row_bytes = (ulong)blocks_per_row * Q1GU_BLOCK_BYTES;
    const ulong d_off = (ulong)blocks_per_row * 16u;
    device const uchar *grb[4];
    device const uchar *urb[4];
    for (uint r = 0; r < ROWS_PER_SG_GU; ++r) {
        const ulong off = (ulong)min(row0 + r, N - 1u) * row_bytes;
        grb[r] = gate_packed + off;
        urb[r] = up_packed + off;
    }
    device const float4 *x16 = reinterpret_cast<device const float4*>(x);

    float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float uacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint blk = simd_lane_id; blk < blocks_per_row; blk += SIMDGROUP_SIZE_GU) {
        float gd[4], ud[4];
        ushort gw[4][8], uw[4][8];
        for (uint r = 0; r < ROWS_PER_SG_GU; ++r) {
            const uint4 gs4 =
                *reinterpret_cast<device const uint4*>(grb[r] + (ulong)blk * 16u);
            const uint4 us4 =
                *reinterpret_cast<device const uint4*>(urb[r] + (ulong)blk * 16u);
            gw[r][0] = ushort(gs4.x & 0xFFFFu); gw[r][1] = ushort(gs4.x >> 16);
            gw[r][2] = ushort(gs4.y & 0xFFFFu); gw[r][3] = ushort(gs4.y >> 16);
            gw[r][4] = ushort(gs4.z & 0xFFFFu); gw[r][5] = ushort(gs4.z >> 16);
            gw[r][6] = ushort(gs4.w & 0xFFFFu); gw[r][7] = ushort(gs4.w >> 16);
            uw[r][0] = ushort(us4.x & 0xFFFFu); uw[r][1] = ushort(us4.x >> 16);
            uw[r][2] = ushort(us4.y & 0xFFFFu); uw[r][3] = ushort(us4.y >> 16);
            uw[r][4] = ushort(us4.z & 0xFFFFu); uw[r][5] = ushort(us4.z >> 16);
            uw[r][6] = ushort(us4.w & 0xFFFFu); uw[r][7] = ushort(us4.w >> 16);
            gd[r] = float(*reinterpret_cast<device const half*>(
                grb[r] + d_off + (ulong)blk * 2u));
            ud[r] = float(*reinterpret_cast<device const half*>(
                urb[r] + d_off + (ulong)blk * 2u));
        }

        float gs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float us[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint w = 0; w < Q1GU_WORDS_PER_BLOCK; ++w) {
            const float4 lo = x16[blk * 16u + w * 2u];
            const float4 hi = x16[blk * 16u + w * 2u + 1u];
            const bfloat4 xa = as_type<bfloat4>(lo.xy);
            const bfloat4 xb = as_type<bfloat4>(lo.zw);
            const bfloat4 xc = as_type<bfloat4>(hi.xy);
            const bfloat4 xd = as_type<bfloat4>(hi.zw);
            const float xv[16] = {
                float(xa.x), float(xa.y), float(xa.z), float(xa.w),
                float(xb.x), float(xb.y), float(xb.z), float(xb.w),
                float(xc.x), float(xc.y), float(xc.z), float(xc.w),
                float(xd.x), float(xd.y), float(xd.z), float(xd.w),
            };
            for (uint r = 0; r < ROWS_PER_SG_GU; ++r) {
                gs[r] += q1gu_dot16f(gw[r][w], xv);
                us[r] += q1gu_dot16f(uw[r][w], xv);
            }
        }
        for (uint r = 0; r < ROWS_PER_SG_GU; ++r) {
            gacc[r] += gd[r] * gs[r];
            uacc[r] += ud[r] * us[r];
        }
    }

    for (uint r = 0; r < ROWS_PER_SG_GU; ++r) {
        const float g_sum = simd_sum(gacc[r]);
        const float u_sum = simd_sum(uacc[r]);
        if (simd_lane_id == 0 && row0 + r < N) {
            gate_y[row0 + r] = bfloat(g_sum);
            up_y[row0 + r] = bfloat(u_sum);
        }
    }
}
