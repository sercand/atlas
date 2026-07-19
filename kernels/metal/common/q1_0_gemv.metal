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
// 16-bit loads — no byte assembly needed (unlike the 34-byte Q2_0
// blocks on CUDA).
//
// Mirrors `mlx_int8_gemv`'s shape: 4 output rows per threadgroup, one
// simdgroup (32 lanes) per row, `simd_sum` reduction, no barriers.
// Each lane iterates 16-bit sign words (16 weights each = 8 sign
// words per block); the per-block fp16 scale is re-read per word
// (L1-resident, negligible next to the x/sign traffic).
//
// Preconditions: K % 128 == 0 (every Bonsai-27B linear: K = 5120,
// 17408, 4096, 6144). Caller dispatches ceil(N/4) threadgroups of 128
// threads.

#include <metal_stdlib>
using namespace metal;

constant uint ROWS_PER_TG    = 4u;
constant uint SIMDGROUP_SIZE = 32u;
constant uint Q1_GROUP       = 128u;
constant uint Q1_BLOCK_BYTES = 18u;
constant uint WORDS_PER_BLOCK = 8u; // 8 x u16 sign words = 128 bits

// Signed accumulate of 16 inputs (four bfloat4s) against one sign word.
static inline float q1_dot16(ushort bits, bfloat4 a, bfloat4 b, bfloat4 c, bfloat4 e)
{
    const float xv[16] = {
        float(a.x), float(a.y), float(a.z), float(a.w),
        float(b.x), float(b.y), float(b.z), float(b.w),
        float(c.x), float(c.y), float(c.z), float(c.w),
        float(e.x), float(e.y), float(e.z), float(e.w),
    };
    float s = 0.0f;
    for (uint i = 0; i < 16; ++i) {
        s += ((bits >> i) & 1u) ? xv[i] : -xv[i];
    }
    return s;
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
    const uint row = tg_idx * ROWS_PER_TG + simd_group_id;
    if (row >= N) {
        return;
    }

    const uint blocks_per_row = K / Q1_GROUP;
    const uint words_per_row  = blocks_per_row * WORDS_PER_BLOCK;
    device const uchar *row_base = packed + (ulong)row * blocks_per_row * Q1_BLOCK_BYTES;
    device const bfloat4 *x4 = reinterpret_cast<device const bfloat4*>(x);

    float acc = 0.0f;
    // Each lane strides over 16-bit sign words (16 weights / word).
    for (uint w = simd_lane_id; w < words_per_row; w += SIMDGROUP_SIZE) {
        const uint blk  = w / WORDS_PER_BLOCK;
        const uint wblk = w % WORDS_PER_BLOCK;
        device const uchar *bp = row_base + blk * Q1_BLOCK_BYTES;
        const float d = float(*reinterpret_cast<device const half*>(bp));
        const ushort bits =
            *reinterpret_cast<device const ushort*>(bp + 2u + wblk * 2u);

        // 16 x values for this word: elements [w*16, w*16+16).
        acc += d * q1_dot16(bits,
                            x4[w * 4u], x4[w * 4u + 1u],
                            x4[w * 4u + 2u], x4[w * 4u + 3u]);
    }

    const float row_sum = simd_sum(acc);
    if (simd_lane_id == 0) {
        y[row] = bfloat(row_sum);
    }
}

// Batched variant: y[m, n] for M co-scheduled decode lanes sharing one
// weight read. Same per-row simdgroup layout; the M x-vectors are
// walked in an inner loop so the packed bytes (the bandwidth term) are
// loaded once per row regardless of M. M is small (<= 8 in practice).
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
    const uint row = tg_idx * ROWS_PER_TG + simd_group_id;
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
