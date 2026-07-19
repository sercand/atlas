// SPDX-License-Identifier: AGPL-3.0-only
//
// Vision-tower 2D RoPE apply — rotate-half with a host-precomputed
// per-patch cos/sin table (the Qwen3-VL `[row_freq;col_freq;row_freq;
// col_freq]` structure lives entirely in that table, so this kernel is
// a plain elementwise rotate-half over the FULL head_dim).
//
//   out[i]        = x[i] * cos[i] - x[i + half] * sin[i]        (i < half)
//   out[i + half] = x[i + half] * cos[i + half] + x[i] * sin[i + half]
//
// cos/sin rows repeat their first half in the second half by
// construction, making the two lines above the standard NeoX rotation.
// Applied in-place, one launch per tensor (Q, then K).
//
// Layout:
//   x   : bfloat [P, num_heads, head_dim]
//   cs  : bfloat [P, head_dim]  cos table
//   sn  : bfloat [P, head_dim]  sin table
//
// Grid: (head_dim/2 threads, num_heads, P)

#include <metal_stdlib>
using namespace metal;

kernel void vision_rope_apply(
    constant uint  &num_tokens [[buffer(0)]],
    constant uint  &num_heads  [[buffer(1)]],
    constant uint  &head_dim   [[buffer(2)]],
    device const bfloat *cs [[buffer(3)]],
    device const bfloat *sn [[buffer(4)]],
    device bfloat       *x  [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint d   = gid.x;         // 0 .. head_dim/2
    uint h   = gid.y;
    uint tok = gid.z;
    uint half_dim = head_dim >> 1u;
    if (d >= half_dim || h >= num_heads || tok >= num_tokens) {
        return;
    }

    uint trow = tok * head_dim;
    float c_lo = float(cs[trow + d]);
    float s_lo = float(sn[trow + d]);
    float c_hi = float(cs[trow + d + half_dim]);
    float s_hi = float(sn[trow + d + half_dim]);

    uint base = (tok * num_heads + h) * head_dim;
    uint i_lo = base + d;
    uint i_hi = base + d + half_dim;
    float lo = float(x[i_lo]);
    float hi = float(x[i_hi]);
    x[i_lo] = bfloat(lo * c_lo - hi * s_lo);
    x[i_hi] = bfloat(hi * c_hi + lo * s_hi);
}
