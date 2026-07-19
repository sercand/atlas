// SPDX-License-Identifier: AGPL-3.0-only
//
// Interleaved MRoPE (Qwen3.5/3.6-VL) — 3-component rotary position
// embedding, applied in-place to a Q or K tensor. Metal port of
// `kernels/gb10/common/rope_mrope_interleaved.cu`.
//
// Identical math to `rope_apply` (GPT-NeoX rotate-half pairs
// `(d, d + rotary_dim/2)`, shared freq schedule
// `inv_freq[d] = 1/theta^(2d/rotary_dim)`) except the POSITION for
// each rotary pair comes from one of three per-token streams,
// selected round-robin by `d % 3`:
//
//   0 → temporal (t), 1 → height (h), 2 → width (w)
//
// For text tokens t == h == w, which reproduces `rope_apply`
// bit-for-bit; the streams only diverge across image-patch runs.
// `mrope_section = [11,11,10]` is a consistency invariant of this
// interleaving ((11+11+10)*2 == rotary_dim == 64), not a runtime
// input.
//
// Layout:
//   x         : bfloat [num_tokens, num_heads, head_dim]
//   inv_freq  : float  [rotary_dim / 2]
//   positions : uint32 [num_tokens, 3]  (t, h, w per token)
//
// Grid: (rotary_dim/2 threads, num_heads, num_tokens)

#include <metal_stdlib>
using namespace metal;

kernel void rope_mrope_interleaved(
    constant uint  &num_tokens [[buffer(0)]],
    constant uint  &num_heads  [[buffer(1)]],
    constant uint  &head_dim   [[buffer(2)]],
    constant uint  &rotary_dim [[buffer(3)]],
    device const uint   *positions [[buffer(4)]], // [num_tokens, 3]
    device const float  *inv_freq  [[buffer(5)]],
    device bfloat       *x         [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint d   = gid.x;         // rotary pair index, 0 .. rotary_dim/2
    uint h   = gid.y;         // head index
    uint tok = gid.z;         // token index
    uint half_rot = rotary_dim >> 1u;
    if (d >= half_rot || h >= num_heads || tok >= num_tokens) {
        return;
    }

    // Round-robin section ownership: pair d belongs to t/h/w by d % 3.
    uint pos = positions[tok * 3u + (d % 3u)];
    float theta = float(pos) * inv_freq[d];
    float c = cos(theta);
    float s = sin(theta);

    uint base = (tok * num_heads + h) * head_dim;
    uint i_lo = base + d;
    uint i_hi = base + d + half_rot;
    float lo = float(x[i_lo]);
    float hi = float(x[i_hi]);
    x[i_lo] = bfloat(lo * c - hi * s);
    x[i_hi] = bfloat(lo * s + hi * c);
}
