// SPDX-License-Identifier: AGPL-3.0-only
//
// Decode-path GEMV over keep-packed PrismML Q1_0 (ggml id 41) weights:
//
//   y[n] = sum_k W_dequant[n, k] * x[k]
//
// Q1_0 block (fixed group 128, 18 bytes, 2-byte aligned):
//   [ fp16 d ][ 16 bytes of sign bits, LSB-first ]
//   value(j) = bit(qs[j/8] >> (j%8)) ? +d : -d
// Row n of a [N, K] weight = K/128 contiguous blocks starting at byte
// offset n * (K/128) * 18. Every block starts on an even byte (18 is
// even), so the fp16 scale and the sign words can be read as aligned
// 16-bit loads.
//
// Layout: 128-thread threadgroups = 4 simdgroups, each simdgroup
// owning FOUR consecutive rows (ROWS_PER_SG) → 16 rows per
// threadgroup, grid ceil(N/16), WORD-STRIDED lanes — see
// `q1_rows4_accum` below for why this exact shape feeds the memory
// pipe (decode streams ~3.4 GB of packed weights per generated token).
//
// Preconditions: K % 128 == 0 (every Bonsai-27B call site: K = 5120,
// 17408, 4096, 6144). N need not be a multiple of the row group —
// tail rows clamp their reads and guard their writes.

#include <metal_stdlib>
using namespace metal;

constant uint SIMDGROUP_SIZE = 32u;
constant uint Q1_GROUP       = 128u;
constant uint Q1_BLOCK_BYTES = 18u;
constant uint WORDS_PER_BLOCK = 8u; // 8 x u16 sign words = 128 bits
constant uint ROWS_PER_SG    = 4u;
constant uint ROWS_PER_TG    = 16u; // 4 simdgroups x 4 rows

// Signed accumulate of 16 already-converted inputs against one sign word.
static inline float q1_dot16f(ushort bits, thread const float *xv)
{
    float s = 0.0f;
    for (uint i = 0; i < 16; ++i) {
        s += ((bits >> i) & 1u) ? xv[i] : -xv[i];
    }
    return s;
}

// Signed accumulate of 16 inputs (four bfloat4s) against one sign word.
// Kept for the batched variant below.
static inline float q1_dot16(ushort bits, bfloat4 a, bfloat4 b, bfloat4 c, bfloat4 e)
{
    const float xv[16] = {
        float(a.x), float(a.y), float(a.z), float(a.w),
        float(b.x), float(b.y), float(b.z), float(b.w),
        float(c.x), float(c.y), float(c.z), float(c.w),
        float(e.x), float(e.y), float(e.z), float(e.w),
    };
    return q1_dot16f(bits, xv);
}

// Four-row simdgroup accumulator, WORD-STRIDED: lane `l` owns 16-bit
// sign word `l, l+32, …` of each of its simdgroup's four rows.
// Adjacent lanes touch adjacent bytes in all four sign streams AND in
// x (fully coalesced), while the four row streams amortize each x
// load + bf16→f32 convert 4×. (A block-per-lane variant was tried and
// lost ~15%: every lane streaming its own 256-byte x window turns
// each x load instruction into a 32-line gather.) The per-block fp16
// scale is re-read per word — it rides the same cache line as the
// signs. `row0` is the first row; rows past N-1 clamp to row N-1
// (valid reads, discarded writes).
static inline void q1_rows4_accum(device const uchar *packed,
                                  uint N,
                                  uint blocks_per_row,
                                  uint row0,
                                  device const bfloat4 *x4,
                                  uint lane,
                                  thread float acc[4])
{
    const ulong row_bytes = (ulong)blocks_per_row * Q1_BLOCK_BYTES;
    device const uchar *rb[4];
    for (uint r = 0; r < ROWS_PER_SG; ++r) {
        rb[r] = packed + (ulong)min(row0 + r, N - 1u) * row_bytes;
    }

    const uint words_per_row = blocks_per_row * WORDS_PER_BLOCK;
    for (uint w = lane; w < words_per_row; w += SIMDGROUP_SIZE) {
        const uint boff = (w / WORDS_PER_BLOCK) * Q1_BLOCK_BYTES;
        const uint soff = boff + 2u + (w % WORDS_PER_BLOCK) * 2u;

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
            const float d =
                float(*reinterpret_cast<device const half*>(rb[r] + boff));
            const ushort bits =
                *reinterpret_cast<device const ushort*>(rb[r] + soff);
            acc[r] += d * q1_dot16f(bits, xv);
        }
    }
}

kernel void q1_0_gemv(
    constant uint &N            [[buffer(0)]],
    constant uint &K            [[buffer(1)]],
    device const uchar  *packed [[buffer(2)]],
    device const bfloat *x      [[buffer(3)]],
    device bfloat       *y      [[buffer(4)]],
    uint   tg_idx        [[threadgroup_position_in_grid]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    const uint row0 = tg_idx * ROWS_PER_TG + simd_group_id * ROWS_PER_SG;
    if (row0 >= N) {
        return;
    }
    const uint blocks_per_row = K / Q1_GROUP;
    device const bfloat4 *x4 = reinterpret_cast<device const bfloat4*>(x);

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    q1_rows4_accum(packed, N, blocks_per_row, row0, x4, simd_lane_id, acc);

    for (uint r = 0; r < ROWS_PER_SG; ++r) {
        const float row_sum = simd_sum(acc[r]);
        if (simd_lane_id == 0 && row0 + r < N) {
            y[row0 + r] = bfloat(row_sum);
        }
    }
}

// Plain GEMV + residual-stream addition:
//   y[n] = x_resid[n] + sum_k W[n, k] * x[k]
// Pairs with the elementwise `silu_gate` kernel to replace the fused
// silu-in-gemv path (which recomputed the activation once per output
// row — N times instead of once).
kernel void q1_0_gemv_resid(
    constant uint &N             [[buffer(0)]],
    constant uint &K             [[buffer(1)]],
    device const uchar  *packed  [[buffer(2)]],
    device const bfloat *x       [[buffer(3)]],
    device const bfloat *x_resid [[buffer(4)]],
    device bfloat       *y       [[buffer(5)]],
    uint   tg_idx        [[threadgroup_position_in_grid]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    const uint row0 = tg_idx * ROWS_PER_TG + simd_group_id * ROWS_PER_SG;
    if (row0 >= N) {
        return;
    }
    const uint blocks_per_row = K / Q1_GROUP;
    device const bfloat4 *x4 = reinterpret_cast<device const bfloat4*>(x);

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    q1_rows4_accum(packed, N, blocks_per_row, row0, x4, simd_lane_id, acc);

    for (uint r = 0; r < ROWS_PER_SG; ++r) {
        const float row_sum = simd_sum(acc[r]);
        if (simd_lane_id == 0 && row0 + r < N) {
            y[row0 + r] = bfloat(row_sum + float(x_resid[row0 + r]));
        }
    }
}

// ── Row-planar variants ─────────────────────────────────────────────
//
// The loader's `PackedQ1Planar` layout puts, per row, all 16-byte sign
// runs first (one aligned uint4 per block) followed by all fp16 scales:
//   [ s_0 … s_{b-1} | d_0 … d_{b-1} ]   (same 18·b bytes as block order)
// Row starts are 16-byte aligned (b % 8 == 0 gate in the loader), so a
// lane loads a block's 128 sign bits as ONE aligned uint4 and its scale
// as a coalesced half — the misaligned 18-byte block stride disappears.

static inline void q1_rows4_accum_planar(device const uchar *packed,
                                         uint N,
                                         uint blocks_per_row,
                                         uint row0,
                                         device const float4 *x16,
                                         uint lane,
                                         thread float acc[4])
{
    const ulong row_bytes = (ulong)blocks_per_row * Q1_BLOCK_BYTES;
    const ulong d_off = (ulong)blocks_per_row * 16u; // sign plane size
    device const uchar *rb[4];
    for (uint r = 0; r < ROWS_PER_SG; ++r) {
        rb[r] = packed + (ulong)min(row0 + r, N - 1u) * row_bytes;
    }

    for (uint blk = lane; blk < blocks_per_row; blk += SIMDGROUP_SIZE) {
        float d[4];
        ushort sw[4][8];
        for (uint r = 0; r < ROWS_PER_SG; ++r) {
            const uint4 s =
                *reinterpret_cast<device const uint4*>(rb[r] + (ulong)blk * 16u);
            sw[r][0] = ushort(s.x & 0xFFFFu); sw[r][1] = ushort(s.x >> 16);
            sw[r][2] = ushort(s.y & 0xFFFFu); sw[r][3] = ushort(s.y >> 16);
            sw[r][4] = ushort(s.z & 0xFFFFu); sw[r][5] = ushort(s.z >> 16);
            sw[r][6] = ushort(s.w & 0xFFFFu); sw[r][7] = ushort(s.w >> 16);
            d[r] = float(*reinterpret_cast<device const half*>(
                rb[r] + d_off + (ulong)blk * 2u));
        }

        float bsum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint w = 0; w < WORDS_PER_BLOCK; ++w) {
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
            for (uint r = 0; r < ROWS_PER_SG; ++r) {
                bsum[r] += q1_dot16f(sw[r][w], xv);
            }
        }
        for (uint r = 0; r < ROWS_PER_SG; ++r) {
            acc[r] += d[r] * bsum[r];
        }
    }
}

kernel void q1_0_gemv_planar(
    constant uint &N            [[buffer(0)]],
    constant uint &K            [[buffer(1)]],
    device const uchar  *packed [[buffer(2)]],
    device const bfloat *x      [[buffer(3)]],
    device bfloat       *y      [[buffer(4)]],
    uint   tg_idx        [[threadgroup_position_in_grid]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    const uint row0 = tg_idx * ROWS_PER_TG + simd_group_id * ROWS_PER_SG;
    if (row0 >= N) {
        return;
    }
    device const float4 *x16 = reinterpret_cast<device const float4*>(x);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    q1_rows4_accum_planar(packed, N, K / Q1_GROUP, row0, x16, simd_lane_id, acc);

    for (uint r = 0; r < ROWS_PER_SG; ++r) {
        const float row_sum = simd_sum(acc[r]);
        if (simd_lane_id == 0 && row0 + r < N) {
            y[row0 + r] = bfloat(row_sum);
        }
    }
}

kernel void q1_0_gemv_resid_planar(
    constant uint &N             [[buffer(0)]],
    constant uint &K             [[buffer(1)]],
    device const uchar  *packed  [[buffer(2)]],
    device const bfloat *x       [[buffer(3)]],
    device const bfloat *x_resid [[buffer(4)]],
    device bfloat       *y       [[buffer(5)]],
    uint   tg_idx        [[threadgroup_position_in_grid]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    const uint row0 = tg_idx * ROWS_PER_TG + simd_group_id * ROWS_PER_SG;
    if (row0 >= N) {
        return;
    }
    device const float4 *x16 = reinterpret_cast<device const float4*>(x);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    q1_rows4_accum_planar(packed, N, K / Q1_GROUP, row0, x16, simd_lane_id, acc);

    for (uint r = 0; r < ROWS_PER_SG; ++r) {
        const float row_sum = simd_sum(acc[r]);
        if (simd_lane_id == 0 && row0 + r < N) {
            y[row0 + r] = bfloat(row_sum + float(x_resid[row0 + r]));
        }
    }
}

// Batched variant: y[m, n] for M co-scheduled decode lanes sharing one
// weight read. Legacy 4-rows-per-threadgroup layout (one simdgroup per
// row), grid ceil(N/4) — see GgufQ1Weight::gemv_batchm. The M
// x-vectors are walked in an inner loop so the packed bytes (the
// bandwidth term) are loaded once per row regardless of M (M <= 8).
kernel void q1_0_gemv_batchm(
    constant uint &N            [[buffer(0)]],
    constant uint &K            [[buffer(1)]],
    constant uint &M            [[buffer(2)]],
    device const uchar  *packed [[buffer(3)]],
    device const bfloat *x      [[buffer(4)]], // [M, K]
    device bfloat       *y      [[buffer(5)]], // [M, N]
    uint   tg_idx        [[threadgroup_position_in_grid]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    const uint row = tg_idx * 4u + simd_group_id;
    if (row >= N) {
        return;
    }

    const uint blocks_per_row = K / Q1_GROUP;
    const uint words_per_row  = blocks_per_row * WORDS_PER_BLOCK;
    device const uchar *row_base = packed + (ulong)row * blocks_per_row * Q1_BLOCK_BYTES;

    float acc[8] = {0.0f};
    for (uint w = simd_lane_id; w < words_per_row; w += SIMDGROUP_SIZE) {
        const uint blk  = w / WORDS_PER_BLOCK;
        const uint wblk = w % WORDS_PER_BLOCK;
        device const uchar *bp = row_base + blk * Q1_BLOCK_BYTES;
        const float d = float(*reinterpret_cast<device const half*>(bp));
        const ushort bits =
            *reinterpret_cast<device const ushort*>(bp + 2u + wblk * 2u);

        for (uint m = 0; m < M; ++m) {
            device const bfloat4 *x4 =
                reinterpret_cast<device const bfloat4*>(x + (ulong)m * K);
            acc[m] += d * q1_dot16(bits,
                                   x4[w * 4u], x4[w * 4u + 1u],
                                   x4[w * 4u + 2u], x4[w * 4u + 3u]);
        }
    }

    for (uint m = 0; m < M; ++m) {
        const float row_sum = simd_sum(acc[m]);
        if (simd_lane_id == 0) {
            y[(ulong)m * N + row] = bfloat(row_sum);
        }
    }
}
