// SPDX-License-Identifier: AGPL-3.0-only
//
// Plain BF16 matmul for unquantized weights:
//
//   y[m, n] = sum_k(x[m, k] * w[n, k])
//
// `w` is laid out `[N, K]` (out-features × in-features), the same
// row-major layout as `dense_gemv_bf16`. Used for ViT-style prefill
// where every patch token needs the same linear projection.
//
// One thread per (m, n) output cell — straightforward correctness
// reference; tile-optimised replacement (simdgroup_matrix) is a
// follow-on PR. The call shape stays stable so callers are unaffected.
//
// Layout:
//   x : bfloat [M, K]
//   w : bfloat [N, K]
//   y : bfloat [M, N]

#include <metal_stdlib>
using namespace metal;

kernel void dense_gemm_bf16(
    constant uint &M  [[buffer(0)]],
    constant uint &N  [[buffer(1)]],
    constant uint &K  [[buffer(2)]],
    device const bfloat *x [[buffer(3)]],
    device const bfloat *w [[buffer(4)]],
    device bfloat       *y [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint m = gid.y;
    uint n = gid.x;
    if (m >= M || n >= N) {
        return;
    }
    float acc = 0.0f;
    if ((K & 3u) == 0u) {
        // K % 4 == 0 keeps every row bfloat4-aligned → vector loads.
        device const bfloat4 *x4 = reinterpret_cast<device const bfloat4*>(x + m * K);
        device const bfloat4 *w4 = reinterpret_cast<device const bfloat4*>(w + n * K);
        for (uint k4 = 0; k4 < K / 4u; ++k4) {
            const bfloat4 a = x4[k4];
            const bfloat4 b = w4[k4];
            acc += float(a.x) * float(b.x) + float(a.y) * float(b.y)
                 + float(a.z) * float(b.z) + float(a.w) * float(b.w);
        }
    } else {
        for (uint k = 0; k < K; ++k) {
            acc += float(x[m * K + k]) * float(w[n * K + k]);
        }
    }
    y[m * N + n] = bfloat(acc);
}
