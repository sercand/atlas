// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the Qwen3.8-Flash-Next QSA indexer kernels
//! (`qsa_indexer.cu`): block-key pooling, decode-query prep, block scoring
//! and the selected-token K/V gather. See the .cu header for the semantics
//! and the scratch-as-paged-cache trick that lets the EXISTING paged decode
//! attention consume the selection.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Pool `n_new` freshly complete blocks starting at `first_block`:
/// mean over `ratio` raw keys -> RMSNorm*(1+w) -> rope at block-start pos.
#[allow(clippy::too_many_arguments)]
pub fn qsa_block_pool(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    raw_keys: DevicePtr,
    k_norm_w: DevicePtr,
    block_keys: DevicePtr,
    first_block: u32,
    n_new: u32,
    ratio: u32,
    hd: u32,
    rot: u32,
    theta: f32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    if n_new == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel)
        .grid([n_new, 1, 1])
        .block([hd, 1, 1])
        .shared_mem((hd + 32) * 4)
        .arg_ptr(raw_keys)
        .arg_ptr(k_norm_w)
        .arg_ptr(block_keys)
        .arg_u32(first_block)
        .arg_u32(ratio)
        .arg_u32(hd)
        .arg_u32(rot)
        .arg_f32(theta)
        .arg_f32(eps)
        .launch(stream)
}

/// One decode query: per head, RMSNorm*(1+w) + partial rope at `pos` -> FP32.
#[allow(clippy::too_many_arguments)]
pub fn qsa_qprep(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_in: DevicePtr,
    q_norm_w: DevicePtr,
    q_out: DevicePtr,
    n_heads: u32,
    hd: u32,
    rot: u32,
    pos: u32,
    theta: f32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n_heads, 1, 1])
        .block([hd, 1, 1])
        .shared_mem((hd + 32) * 4)
        .arg_ptr(q_in)
        .arg_ptr(q_norm_w)
        .arg_ptr(q_out)
        .arg_u32(hd)
        .arg_u32(rot)
        .arg_u32(pos)
        .arg_f32(theta)
        .arg_f32(eps)
        .launch(stream)
}

/// scores[b] = sum_h relu(q_h . k_b) / sqrt(hd) over `n_blocks` blocks.
#[allow(clippy::too_many_arguments)]
pub fn qsa_score(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    block_keys: DevicePtr,
    scores: DevicePtr,
    n_blocks: u32,
    n_heads: u32,
    hd: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n_blocks, 1, 1])
        .block([hd, 1, 1])
        .shared_mem(32 * 4)
        .arg_ptr(q)
        .arg_ptr(block_keys)
        .arg_ptr(scores)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .launch(stream)
}

/// Pack the selected tokens' K/V rows into contiguous NHD scratch.
#[allow(clippy::too_many_arguments)]
pub fn qsa_gather(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    block_table: DevicePtr,
    sel: DevicePtr,
    k_out: DevicePtr,
    v_out: DevicePtr,
    n_sel: u32,
    block_size: u32,
    nkv: u32,
    hd: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n_sel, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(block_table)
        .arg_ptr(sel)
        .arg_ptr(k_out)
        .arg_ptr(v_out)
        .arg_u32(block_size)
        .arg_u32(nkv)
        .arg_u32(hd)
        .launch(stream)
}

/// Stage 2: per-row q prep for a contiguous selective row range.
#[allow(clippy::too_many_arguments)]
pub fn qsa_qprep_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    qk: DevicePtr,
    q_norm_w: DevicePtr,
    q_out: DevicePtr,
    rows: u32,
    first_pos: u32,
    qkw: u32,
    n_heads: u32,
    hd: u32,
    rot: u32,
    theta: f32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, n_heads, 1])
        .block([hd, 1, 1])
        .shared_mem((hd + 32) * 4)
        .arg_ptr(qk)
        .arg_ptr(q_norm_w)
        .arg_ptr(q_out)
        .arg_u32(first_pos)
        .arg_u32(qkw)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .arg_u32(rot)
        .arg_f32(theta)
        .arg_f32(eps)
        .launch(stream)
}

/// Stage 2: per-row block scores, -inf beyond each row's complete count.
#[allow(clippy::too_many_arguments)]
pub fn qsa_score_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    block_keys: DevicePtr,
    scores: DevicePtr,
    rows: u32,
    n_blocks_max: u32,
    first_pos: u32,
    score_stride: u32,
    ratio: u32,
    n_heads: u32,
    hd: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, n_blocks_max, 1])
        .block([hd, 1, 1])
        .shared_mem(32 * 4)
        .arg_ptr(q)
        .arg_ptr(block_keys)
        .arg_ptr(scores)
        .arg_u32(first_pos)
        .arg_u32(score_stride)
        .arg_u32(ratio)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .launch(stream)
}

/// Block-tiled `qsa_score_rows`: one CTA covers `QSA_SCORE_TB` blocks of a row
/// instead of one, staging the row's query tile in shared memory so it is read
/// once per tile rather than once per block. Scores are bit-identical (same
/// reduction tree, same operands, same per-head accumulate order).
///
/// `QSA_SCORE_TB` is 16 and must match the `#define` in `qsa_indexer.cu`.
#[allow(clippy::too_many_arguments)]
pub fn qsa_score_rows_tiled(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    block_keys: DevicePtr,
    scores: DevicePtr,
    rows: u32,
    n_blocks_max: u32,
    first_pos: u32,
    score_stride: u32,
    ratio: u32,
    n_heads: u32,
    hd: u32,
    stream: u64,
) -> Result<()> {
    const TB: u32 = 16;
    KernelLaunch::new(gpu, kernel)
        .grid([rows, n_blocks_max.div_ceil(TB), 1])
        .block([hd, 1, 1])
        // [red(32 floats) | q_tile(n_heads*hd floats)]
        .shared_mem((32 + n_heads * hd) * 4)
        .arg_ptr(q)
        .arg_ptr(block_keys)
        .arg_ptr(scores)
        .arg_u32(first_pos)
        .arg_u32(score_stride)
        .arg_u32(ratio)
        .arg_u32(n_heads)
        .arg_u32(hd)
        .arg_u32(n_blocks_max)
        .launch(stream)
}

/// On-device per-row top-k over the score matrix, writing `lists` directly.
///
/// Replaces the D2H of the whole score matrix + the host quickselect. Output is
/// contractually identical to `host_topk_blocks` — same block ids in the same
/// score-DESC/index-ASC order — because the kernel sorts the same total order
/// expressed as a 64-bit key. See `qsa_topk_rows` in `qsa_indexer.cu`.
///
/// One CTA per row; `topk` must be a power of two (the in-kernel bitonic sort)
/// and no larger than the `lists` row stride.
pub fn qsa_topk_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    scores: DevicePtr,
    lists: DevicePtr,
    rows: u32,
    first_pos: u32,
    score_stride: u32,
    ratio: u32,
    topk: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        topk.is_power_of_two(),
        "qsa_topk_rows: topk={topk} is not a power of two (in-kernel bitonic sort)"
    );
    // topk*8 bytes of key buffer + QSA_TOPK_HIST_U32(260)*4 of histogram.
    let smem = topk * 8 + 260 * 4;
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([256, 1, 1])
        .shared_mem(smem)
        .arg_ptr(scores)
        .arg_ptr(lists)
        .arg_u32(first_pos)
        .arg_u32(score_stride)
        .arg_u32(ratio)
        .arg_u32(topk)
        .launch(stream)
}

/// Stage 2: per-row selected-set attention, overwriting the context rows.
#[allow(clippy::too_many_arguments)]
pub fn qsa_prefill_attn(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    block_table: DevicePtr,
    lists: DevicePtr,
    attn_out: DevicePtr,
    rows: u32,
    first_pos: u32,
    topk: u32,
    ratio: u32,
    block_size: u32,
    nq: u32,
    nkv: u32,
    hd: u32,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    // 8 warps x [hd] acc partials + m/l per warp (slab reused per head).
    let smem = (8 * hd + 16) * 4;
    // Must equal QSA_PA_HPP in qsa_indexer.cu: each CTA runs HPP q-heads of
    // one KV group per K/V pass (the GQA-dedup that fixed the depth-scaled
    // qsa_select wall). One CTA must never straddle KV heads.
    const HPP: u32 = 4;
    anyhow::ensure!(
        nq % HPP == 0 && (nq / nkv) % HPP == 0,
        "qsa_prefill_attn: nq={nq} nkv={nkv} not tileable by HPP={HPP}"
    );
    // The kernel's software-pipelined K/V staging loads each lane's row slice
    // as one uint4 (16 B = 8 bf16), which assumes hd == 256 exactly.
    anyhow::ensure!(hd == 256, "qsa_prefill_attn: uint4 staging requires hd=256, got {hd}");
    KernelLaunch::new(gpu, kernel)
        .grid([rows, nq / HPP, 1])
        .block([256, 1, 1])
        .shared_mem(smem)
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(block_table)
        .arg_ptr(lists)
        .arg_ptr(attn_out)
        .arg_u32(first_pos)
        .arg_u32(topk)
        .arg_u32(ratio)
        .arg_u32(block_size)
        .arg_u32(nq)
        .arg_u32(nkv)
        .arg_u32(hd)
        .arg_f32(inv_sqrt_d)
        .launch(stream)
}
