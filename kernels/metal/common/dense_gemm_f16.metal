// SPDX-License-Identifier: AGPL-3.0-only
//
// Simdgroup-matrix GEMM for dense F16 weights (the vision tower):
//
//   y[t, n] = bias[n] + sum_k W[n, k] * x[t, k]
//
// Same winning shape as `q1_0_gemm` v4 (BM=32 weight rows × BN=128
// tokens × 128 threads; W tile staged K-MAJOR to threadgroup half; A
// fragments simdgroup_load DIRECT from a device-half x image), minus
// the dequant: W is already half, so staging is a coalesced copy with
// row/K-tail zero-fill. BK=64 here — the tower's one non-128-multiple
// K (fc2's 4304) pads to 4352 instead of 4480.
//
// CALLER CONTRACT (see `bf16_to_half_pad`): x is [T_pad, K_pad] with
// T_pad a 128-multiple, K_pad a BK-multiple, and BOTH tails ZERO —
// garbage tail columns would feed NaN×0 into the MMA accumulators.
// Weights convert BF16→F16 in place at init via `bf16_to_half_inplace`.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint DBM = 32u;  // weight rows per threadgroup tile
constant uint DBN = 128u; // tokens per threadgroup tile (32 per simdgroup)
constant uint DBK = 64u;  // k-depth per stage

// In-place BF16 → F16 reinterpret of a weight buffer. Each thread owns
// one element, so aliasing src/dst through the same pointer is safe.
kernel void bf16_to_half_inplace(
    constant uint  &n    [[buffer(0)]],
    device ushort  *buf  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    const bfloat v = as_type<bfloat>(buf[gid]);
    buf[gid] = as_type<ushort>(half(float(v)));
}

// BF16 rows → zero-padded device-half GEMM input image:
//   dst[r, c] = (r < rows && c < k) ? half(src[r, c]) : 0
// dst is [rows_pad, k_pad]; grid covers rows_pad * k_pad.
kernel void bf16_to_half_pad(
    constant uint &rows      [[buffer(0)]],
    constant uint &k         [[buffer(1)]],
    constant uint &rows_pad  [[buffer(2)]],
    constant uint &k_pad     [[buffer(3)]],
    device const bfloat *src [[buffer(4)]],
    device half         *dst [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = rows_pad * k_pad;
    if (gid >= total) {
        return;
    }
    const uint r = gid / k_pad;
    const uint c = gid % k_pad;
    dst[gid] = (r < rows && c < k) ? half(float(src[r * k + c])) : half(0.0h);
}

// Strided variant of `bf16_to_half_pad` for extracting a HEAD SLICE of
// an interleaved `[rows, heads, head_dim]` activation into a padded
// half image (caller pre-offsets `src` to the head):
//   dst[r, c] = (r < rows && c < k) ? half(src[r * src_stride + c]) : 0
kernel void bf16_to_half_pad_strided(
    constant uint &rows       [[buffer(0)]],
    constant uint &k          [[buffer(1)]],
    constant uint &rows_pad   [[buffer(2)]],
    constant uint &k_pad      [[buffer(3)]],
    constant uint &src_stride [[buffer(4)]],
    device const bfloat *src  [[buffer(5)]],
    device half         *dst  [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = rows_pad * k_pad;
    if (gid >= total) {
        return;
    }
    const uint r = gid / k_pad;
    const uint c = gid % k_pad;
    dst[gid] = (r < rows && c < k) ? half(float(src[r * src_stride + c])) : half(0.0h);
}

// Row softmax for the GEMM-attention path: raw f32 scores → half probs
// in a zero-padded image ready to be the next GEMM's A operand.
//   probs[t, s] = exp(scale·x[t,s] − max_s) / Σ exp(scale·x[t,s] − max_s)
// One threadgroup per output row; rows ≥ T and columns ≥ s_len write 0
// (the P·V GEMM's B rows past s_len are zero-padded too, so padding
// contributes exactly nothing). Row staged in threadgroup memory —
// scores are read from device once. s_len ≤ 4096.
kernel void vit_softmax_half(
    constant uint  &t_rows  [[buffer(0)]], // logical rows (queries)
    constant uint  &s_len   [[buffer(1)]], // logical cols (keys)
    constant uint  &s_pad   [[buffer(2)]], // out row stride
    constant float &scale   [[buffer(3)]],
    device const float *x   [[buffer(4)]], // [t_rows, s_len]
    device half        *out [[buffer(5)]], // [t_pad, s_pad]
    uint tg_idx [[threadgroup_position_in_grid]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_grp  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float rowbuf[4096];
    threadgroup float red[32];
    threadgroup float row_max;
    threadgroup float row_sum;

    const uint t = tg_idx;
    if (t >= t_rows) {
        for (uint s = tid; s < s_pad; s += tg_size) {
            out[(ulong)t * s_pad + s] = half(0.0h);
        }
        return;
    }
    const uint num_simds = (tg_size + 31u) / 32u;

    float local_max = -INFINITY;
    for (uint s = tid; s < s_len && s < 4096u; s += tg_size) {
        const float v = scale * x[(ulong)t * s_len + s];
        rowbuf[s] = v;
        local_max = max(local_max, v);
    }
    float m = simd_max(local_max);
    if (simd_lane == 0) {
        red[simd_grp] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_grp == 0) {
        float v0 = (tid < num_simds) ? red[tid] : -INFINITY;
        v0 = simd_max(v0);
        if (tid == 0) {
            row_max = v0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint s = tid; s < s_len; s += tg_size) {
        const float e = exp(rowbuf[s] - row_max);
        rowbuf[s] = e;
        local_sum += e;
    }
    float sm = simd_sum(local_sum);
    if (simd_lane == 0) {
        red[simd_grp] = sm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_grp == 0) {
        float v0 = (tid < num_simds) ? red[tid] : 0.0f;
        v0 = simd_sum(v0);
        if (tid == 0) {
            row_sum = v0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv = 1.0f / row_sum;
    for (uint s = tid; s < s_pad; s += tg_size) {
        out[(ulong)t * s_pad + s] = (s < s_len) ? half(rowbuf[s] * inv) : half(0.0h);
    }
}

// Stage the [DBM, DBK] weight tile at (row0, k0) K-MAJOR into `wt`
// ([DBK][DBM], row stride DBM). Rows past N clamp to N-1 (stores are
// guarded); k past K zero-fills (pairs with x's zero tail columns).
static inline void stage_weights_f16(device const half *w,
                                     threadgroup half  *wt,
                                     uint N, uint K,
                                     uint row0, uint k0,
                                     uint tid)
{
    for (uint idx = tid; idx < DBM * DBK; idx += 128u) {
        const uint r = idx / DBK;
        const uint kk = idx % DBK;
        const uint row = min(row0 + r, N - 1u);
        const uint kcol = k0 + kk;
        wt[kk * DBM + r] = (kcol < K) ? w[(ulong)row * K + kcol] : half(0.0h);
    }
}

kernel void dense_gemm_f16_bias(
    constant uint &N             [[buffer(0)]],
    constant uint &K             [[buffer(1)]], // logical K (un-padded)
    constant uint &T             [[buffer(2)]], // logical tokens
    constant uint &k_pad         [[buffer(3)]], // x row stride, DBK-multiple
    device const half   *w       [[buffer(4)]], // [N, K] row-major
    device const half   *x       [[buffer(5)]], // [T_pad, k_pad], tails zero
    device const bfloat *bias    [[buffer(6)]], // [N]
    device bfloat       *y       [[buffer(7)]], // [T, N]
    uint tg_idx [[threadgroup_position_in_grid]],
    uint tid    [[thread_position_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    threadgroup half wt[DBK * DBM];
    threadgroup float ct[4 * 64];

    const uint row_tiles = (N + DBM - 1u) / DBM;
    const uint row0 = (tg_idx % row_tiles) * DBM;
    const uint t0 = (tg_idx / row_tiles) * DBN;

    simdgroup_float8x8 C[4][4];
    for (uint i = 0; i < 4u; ++i) {
        for (uint j = 0; j < 4u; ++j) {
            C[i][j] = simdgroup_float8x8(0.0f);
        }
    }

    device const half *xs = x + (ulong)(t0 + sgid * 32u) * k_pad;
    const uint kblocks = k_pad / DBK;
    for (uint kb = 0; kb < kblocks; ++kb) {
        stage_weights_f16(w, wt, N, K, row0, kb * DBK, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k8 = 0; k8 < DBK / 8u; ++k8) {
            device const half *xk = xs + kb * DBK + k8 * 8u;
            simdgroup_half8x8 a[4];
            simdgroup_load(a[0], xk, k_pad);
            simdgroup_load(a[1], xk + 8u * k_pad, k_pad);
            simdgroup_load(a[2], xk + 16u * k_pad, k_pad);
            simdgroup_load(a[3], xk + 24u * k_pad, k_pad);
            for (uint j = 0; j < 4u; ++j) {
                simdgroup_half8x8 b;
                simdgroup_load(b, wt + (k8 * 8u) * DBM + j * 8u, DBM);
                simdgroup_multiply_accumulate(C[0][j], a[0], b, C[0][j]);
                simdgroup_multiply_accumulate(C[1][j], a[1], b, C[1][j]);
                simdgroup_multiply_accumulate(C[2][j], a[2], b, C[2][j]);
                simdgroup_multiply_accumulate(C[3][j], a[3], b, C[3][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Guarded store through a per-simdgroup staging patch, + bias.
    threadgroup float *sg_ct = ct + sgid * 64u;
    const uint t0_sg = t0 + sgid * 32u;
    for (uint i = 0; i < 4u; ++i) {
        for (uint j = 0; j < 4u; ++j) {
            simdgroup_store(C[i][j], sg_ct, 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64u; e += 32u) {
                const uint tr = t0_sg + i * 8u + e / 8u;
                const uint nc = row0 + j * 8u + e % 8u;
                if (tr < T && nc < N) {
                    y[(ulong)tr * N + nc] = bfloat(sg_ct[e] + float(bias[nc]));
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// GEMM-attention stage 1 (scores): same tiled shape as
// `dense_gemm_f16_bias` but F32 output and no bias — softmax reads raw
// scores at full precision (bf16-rounded scores would put ~1% noise on
// every attention weight after exp).
kernel void dense_gemm_f16_f32out(
    constant uint &N             [[buffer(0)]],
    constant uint &K             [[buffer(1)]],
    constant uint &T             [[buffer(2)]],
    constant uint &k_pad         [[buffer(3)]],
    device const half   *w       [[buffer(4)]], // [N, K] row-major
    device const half   *x       [[buffer(5)]], // [T_pad, k_pad], tails zero
    device float        *y       [[buffer(6)]], // [T, N]
    uint tg_idx [[threadgroup_position_in_grid]],
    uint tid    [[thread_position_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    threadgroup half wt[DBK * DBM];
    threadgroup float ct[4 * 64];

    const uint row_tiles = (N + DBM - 1u) / DBM;
    const uint row0 = (tg_idx % row_tiles) * DBM;
    const uint t0 = (tg_idx / row_tiles) * DBN;

    simdgroup_float8x8 C[4][4];
    for (uint i = 0; i < 4u; ++i) {
        for (uint j = 0; j < 4u; ++j) {
            C[i][j] = simdgroup_float8x8(0.0f);
        }
    }
    device const half *xs = x + (ulong)(t0 + sgid * 32u) * k_pad;
    const uint kblocks = k_pad / DBK;
    for (uint kb = 0; kb < kblocks; ++kb) {
        stage_weights_f16(w, wt, N, K, row0, kb * DBK, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k8 = 0; k8 < DBK / 8u; ++k8) {
            device const half *xk = xs + kb * DBK + k8 * 8u;
            simdgroup_half8x8 a[4];
            simdgroup_load(a[0], xk, k_pad);
            simdgroup_load(a[1], xk + 8u * k_pad, k_pad);
            simdgroup_load(a[2], xk + 16u * k_pad, k_pad);
            simdgroup_load(a[3], xk + 24u * k_pad, k_pad);
            for (uint j = 0; j < 4u; ++j) {
                simdgroup_half8x8 b;
                simdgroup_load(b, wt + (k8 * 8u) * DBM + j * 8u, DBM);
                simdgroup_multiply_accumulate(C[0][j], a[0], b, C[0][j]);
                simdgroup_multiply_accumulate(C[1][j], a[1], b, C[1][j]);
                simdgroup_multiply_accumulate(C[2][j], a[2], b, C[2][j]);
                simdgroup_multiply_accumulate(C[3][j], a[3], b, C[3][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float *sg_ct = ct + sgid * 64u;
    const uint t0_sg = t0 + sgid * 32u;
    for (uint i = 0; i < 4u; ++i) {
        for (uint j = 0; j < 4u; ++j) {
            simdgroup_store(C[i][j], sg_ct, 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64u; e += 32u) {
                const uint tr = t0_sg + i * 8u + e / 8u;
                const uint nc = row0 + j * 8u + e % 8u;
                if (tr < T && nc < N) {
                    y[(ulong)tr * N + nc] = sg_ct[e];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// GEMM-attention stage 2 (P·V): B is a [K, N] half image read DIRECT
// from device (no transpose, no staging — both operands were written
// zero-padded, so tail fragments contribute exactly 0). The output is
// one head's slice of the interleaved `[T, heads, head_dim]` attention
// buffer, hence the explicit `out_stride` (caller pre-offsets y).
kernel void dense_gemm_f16_nt(
    constant uint &N             [[buffer(0)]], // head_dim (logical)
    constant uint &T             [[buffer(1)]], // queries (logical)
    constant uint &k_pad         [[buffer(2)]], // padded keys (x row stride)
    constant uint &n_pad         [[buffer(3)]], // B row stride
    constant uint &out_stride    [[buffer(4)]], // y row stride (elements)
    device const half   *w       [[buffer(5)]], // [k_pad, n_pad], tails zero
    device const half   *x       [[buffer(6)]], // [T_pad, k_pad], tails zero
    device bfloat       *y       [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint tid    [[thread_position_in_threadgroup]],
    uint sgid   [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    threadgroup float ct[4 * 64];

    const uint row_tiles = (n_pad + DBM - 1u) / DBM;
    const uint row0 = (tg_idx % row_tiles) * DBM;
    const uint t0 = (tg_idx / row_tiles) * DBN;

    simdgroup_float8x8 C[4][4];
    for (uint i = 0; i < 4u; ++i) {
        for (uint j = 0; j < 4u; ++j) {
            C[i][j] = simdgroup_float8x8(0.0f);
        }
    }
    device const half *xs = x + (ulong)(t0 + sgid * 32u) * k_pad;
    const uint kblocks = k_pad / DBK;
    for (uint kb = 0; kb < kblocks; ++kb) {
        for (uint k8 = 0; k8 < DBK / 8u; ++k8) {
            const uint krow = kb * DBK + k8 * 8u;
            device const half *xk = xs + krow;
            simdgroup_half8x8 a[4];
            simdgroup_load(a[0], xk, k_pad);
            simdgroup_load(a[1], xk + 8u * k_pad, k_pad);
            simdgroup_load(a[2], xk + 16u * k_pad, k_pad);
            simdgroup_load(a[3], xk + 24u * k_pad, k_pad);
            for (uint j = 0; j < 4u; ++j) {
                simdgroup_half8x8 b;
                simdgroup_load(b, w + (ulong)krow * n_pad + row0 + j * 8u, n_pad);
                simdgroup_multiply_accumulate(C[0][j], a[0], b, C[0][j]);
                simdgroup_multiply_accumulate(C[1][j], a[1], b, C[1][j]);
                simdgroup_multiply_accumulate(C[2][j], a[2], b, C[2][j]);
                simdgroup_multiply_accumulate(C[3][j], a[3], b, C[3][j]);
            }
        }
    }
    threadgroup float *sg_ct = ct + sgid * 64u;
    const uint t0_sg = t0 + sgid * 32u;
    for (uint i = 0; i < 4u; ++i) {
        for (uint j = 0; j < 4u; ++j) {
            simdgroup_store(C[i][j], sg_ct, 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64u; e += 32u) {
                const uint tr = t0_sg + i * 8u + e / 8u;
                const uint nc = row0 + j * 8u + e % 8u;
                if (tr < T && nc < N) {
                    y[(ulong)tr * out_stride + nc] = bfloat(sg_ct[e]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
