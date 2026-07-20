// SPDX-License-Identifier: AGPL-3.0-only
//
// Prefill-path causal conv1d + SiLU + per-head L2-norm — the token-
// batched counterpart of `causal_conv1d_update_l2norm` (same math per
// token, see that file). The conv window is a sliding read over the
// staged input rows, so every (token, channel) pair is independent:
//
//   in(t)[ch]  = input[t, ch]                       t >= 0
//   in(-m)[ch] = conv_state[ch, d_conv - m]         m = 1 .. d_conv-1
//   acc        = sum_k weight[ch, k] * in(t - (d_conv-1) + k)
//   silu → per-head L2 norm on Q+K channels → output[t, ch]
//
// The kernel READS conv_state (left edge of the first tokens) and does
// NOT write it — `causal_conv1d_prefill_state` runs after and advances
// the state past the chunk. Two kernels because the state write would
// race the other threadgroups' edge reads inside one dispatch.
//
// Layout (batch = 1):
//   conv_state : float  [dim, d_conv]               (read-only here)
//   input      : bfloat [num_tokens, dim]
//   weight     : bfloat [dim, d_conv]
//   output     : bfloat [num_tokens, dim]
//
// Grid: flat `(dim / block_x) * num_tokens` threadgroups; block_x is a
// multiple of head_dim (as the decode variant).

#include <metal_stdlib>
using namespace metal;

constant uint MAX_HEADS_PER_BLOCK_P = 4;
constant uint MAX_SIMDGROUPS_LN_P = 16;

kernel void causal_conv1d_prefill_l2norm(
    device const float  *conv_state [[buffer(0)]],
    device const bfloat *input      [[buffer(1)]],
    device const bfloat *weight     [[buffer(2)]],
    device bfloat       *output     [[buffer(3)]],
    constant uint  &num_tokens   [[buffer(4)]],
    constant uint  &dim          [[buffer(5)]],
    constant uint  &d_conv       [[buffer(6)]],
    constant uint  &qk_channels  [[buffer(7)]],
    constant uint  &head_dim     [[buffer(8)]],
    constant float &l2_eps       [[buffer(9)]],
    uint  tg_idx        [[threadgroup_position_in_grid]],
    uint  tid           [[thread_position_in_threadgroup]],
    uint  tg_size       [[threads_per_threadgroup]],
    uint  simd_lane     [[thread_index_in_simdgroup]],
    uint  simd_grp      [[simdgroup_index_in_threadgroup]])
{
    uint blocks_per_tok = (dim + tg_size - 1) / tg_size;
    uint block_x_idx = tg_idx % blocks_per_tok;
    uint t           = tg_idx / blocks_per_tok;

    uint block_start = block_x_idx * tg_size;
    uint ch = block_start + tid;
    bool block_needs_l2 = (block_start < qk_channels);

    threadgroup float partial[MAX_SIMDGROUPS_LN_P];
    threadgroup float head_inv_norm[MAX_HEADS_PER_BLOCK_P];

    bool valid = (ch < dim && t < num_tokens);
    float silu = 0.0f;

    // ── 1. Causal conv + SiLU ───────────────────────────────────
    if (valid) {
        device const bfloat *w = weight + ch * d_conv;
        float acc = 0.0f;
        for (uint k = 0; k < d_conv; ++k) {
            int src = int(t) + int(k) - int(d_conv - 1u);
            float v = (src >= 0)
                ? float(input[uint(src) * dim + ch])
                // in(-m) lives at conv_state[ch, d_conv - m]:
                // src = -m → index d_conv + src.
                : conv_state[ch * d_conv + uint(int(d_conv) + src)];
            acc += v * float(w[k]);
        }
        float sig = 1.0f / (1.0f + exp(-acc));
        silu = acc * sig;
    }

    // ── 2. Per-head L2 norm for Q+K channels ────────────────────
    if (block_needs_l2) {
        float sq = valid ? (silu * silu) : 0.0f;

        float simd_sum_v = simd_sum(sq);
        if (simd_lane == 0) {
            partial[simd_grp] = simd_sum_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint simds_per_head = head_dim / 32u;
        if (simds_per_head == 0u) simds_per_head = 1u;

        uint head_in_block = tid / head_dim;
        uint pos_in_head = tid % head_dim;
        if (pos_in_head == 0 && head_in_block < MAX_HEADS_PER_BLOCK_P) {
            float total = 0.0f;
            uint base_simd = head_in_block * simds_per_head;
            for (uint i = 0; i < simds_per_head; ++i) {
                total += partial[base_simd + i];
            }
            head_inv_norm[head_in_block] = rsqrt(total + l2_eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            silu *= head_inv_norm[head_in_block];
        }
    }

    if (valid) {
        output[t * dim + ch] = bfloat(silu);
    }
}

// Advance the conv state past the chunk: after tokens 0..T-1, slot k of
// the state must hold in(T - 1 - (d_conv - 1) + k) — from the staged
// input rows when in range, else from the OLD state (short chunks).
// One thread per channel; runs after `causal_conv1d_prefill_l2norm`
// (serial encoder order makes the read-then-write safe).
kernel void causal_conv1d_prefill_state(
    device float        *conv_state [[buffer(0)]],
    device const bfloat *input      [[buffer(1)]],
    constant uint &num_tokens [[buffer(2)]],
    constant uint &dim        [[buffer(3)]],
    constant uint &d_conv     [[buffer(4)]],
    uint ch [[thread_position_in_grid]])
{
    if (ch >= dim) {
        return;
    }
    float next[8]; // d_conv ≤ 8 everywhere (Bonsai: 4)
    for (uint k = 0; k < d_conv; ++k) {
        int src = int(num_tokens) - int(d_conv) + int(k);
        // src < 0 is in(-m) from the old state: in(-m) = state[d_conv - m],
        // so index d_conv + src.
        next[k] = (src >= 0)
            ? float(input[uint(src) * dim + ch])
            : conv_state[ch * d_conv + uint(int(d_conv) + src)];
    }
    for (uint k = 0; k < d_conv; ++k) {
        conv_state[ch * d_conv + k] = next[k];
    }
}
