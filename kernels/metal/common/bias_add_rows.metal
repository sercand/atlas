// SPDX-License-Identifier: AGPL-3.0-only
//
// Broadcast bias add over a row-major [rows, cols] BF16 matrix:
//
//   x[r, c] += bias[c]
//
// In-place; FP32 accumulate. The GEMM kernels are bias-free, so every
// vision-tower linear (QKV/proj/fc1/fc2, patch embed, merger) runs
// this immediately after its dense_gemm_bf16.
//
// Grid: ceil(rows*cols / 256) threadgroups × 256 threads.

#include <metal_stdlib>
using namespace metal;

kernel void bias_add_rows(
    constant uint &rows [[buffer(0)]],
    constant uint &cols [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]],
    device bfloat       *x    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = rows * cols;
    if (gid >= n) {
        return;
    }
    x[gid] = bfloat(float(x[gid]) + float(bias[gid % cols]));
}
