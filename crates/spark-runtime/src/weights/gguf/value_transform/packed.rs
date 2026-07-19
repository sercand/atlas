// SPDX-License-Identifier: AGPL-3.0-only
//! Keep-packed (native-Q1/Q2) byte-level permutations — the value-head
//! reorders applied directly to quantized block bytes so the big GDN
//! projections never expand to BF16. Split from `value_transform.rs`
//! for the 500-LoC cap; the F32 oracle equivalents live there.

use anyhow::{Result, ensure};

use super::GdnDims;

/// For the native keep-packed Q2_0 path (`ATLAS_GGUF_NATIVE_Q2=1`): tensors whose
/// value transform is a *pure whole-row* value-head reorder ([`Op::ReorderRows`]
/// with `head_dim_rows = true`) can stay 2-bit — the permutation moves whole
/// `value_head_dim`-row blocks, and one row is an integer number of `block_q2_0`
/// blocks (`K / group`), so it never splits a group. [`reorder_packed_rows`]
/// applies the SAME permutation directly to the packed bytes, leaving the weight
/// HF-correct while packed. Returns `Some(after_qk)` for the two big GDN input
/// projections (`in_proj_qkv` → V-region only, `in_proj_z` → whole tensor),
/// `None` for everything else (which stays on the dequant→BF16 path):
///   * `in_proj_a`/`in_proj_b`/`dt_bias` reorder SINGLE rows (`head_dim_rows =
///     false`); tiny, left BF16.
///   * `out_proj` is a within-row COLUMN reorder ([`Op::ReorderOutCols`]) that
///     would need an intra-row block shuffle — deferred, stays on its NVFP4/BF16
///     path.
///   * `A_log` / norms are F32 scalar transforms — never packed.
pub fn packed_reorder_rows(hf_name: &str) -> Option<bool> {
    // Match by NAME, not just the `Op`: `conv1d` shares `in_proj_qkv`'s
    // `ReorderRows{after_qk,head_dim_rows}` classification but is a 3-D
    // recurrence tensor (`[d_inner,1,k]`) that must stay BF16 — never packed.
    if hf_name.ends_with(".linear_attn.in_proj_qkv.weight") {
        return Some(true); // reorder V-region rows only (after the Q|K partition)
    }
    if hf_name.ends_with(".linear_attn.in_proj_z.weight") {
        return Some(false); // whole tensor is value heads
    }
    None
}

/// Apply the value-head row reorder ([`packed_reorder_rows`]) directly to the raw
/// `block_q2_0` bytes of a keep-packed tensor. Byte-exact analog of
/// [`reorder_rows`]: because dequant is per-block and one HF row spans
/// `k / group` whole blocks (`row_bytes = (k/group) * block_bytes`), permuting
/// whole `value_head_dim`-row block-runs here is bit-identical to permuting the
/// dequantized rows. `raw` is `[n, k]` row-major packed; returns the reordered
/// copy.
pub fn reorder_packed_rows(
    raw: &[u8],
    hf_shape: &[usize],
    dims: &GdnDims,
    after_qk: bool,
    group: usize,
    block_bytes: usize,
) -> Result<Vec<u8>> {
    ensure!(
        hf_shape.len() == 2,
        "packed reorder expects 2-D [n,k], got {hf_shape:?}"
    );
    let (n, k) = (hf_shape[0], hf_shape[1]);
    ensure!(
        k % group == 0,
        "packed reorder: k={k} not a multiple of group={group}"
    );
    let row_bytes = (k / group) * block_bytes;
    ensure!(
        raw.len() == n * row_bytes,
        "packed reorder: raw len {} != n*row_bytes {}",
        raw.len(),
        n * row_bytes
    );
    let hd = dims.value_head_dim; // rows per value head
    let base_row = if after_qk { dims.qk_rows() } else { 0 };
    let region_rows = dims.num_v_heads * hd;
    ensure!(
        base_row + region_rows <= n,
        "packed reorder: V region rows [{base_row}..{}] exceed n={n}",
        base_row + region_rows
    );
    let mut out = raw.to_vec();
    let blk = hd * row_bytes; // bytes per value head
    for hf in 0..dims.num_v_heads {
        let g = dims.gguf_head(hf);
        let d = (base_row + hf * hd) * row_bytes;
        let s = (base_row + g * hd) * row_bytes;
        out[d..d + blk].copy_from_slice(&raw[s..s + blk]);
    }
    Ok(out)
}

/// For the native keep-packed path: true when this tensor's transform is the
/// `out_proj` within-row COLUMN reorder ([`Op::ReorderOutCols`]), which moves
/// `value_head_dim`-sized column groups. When `value_head_dim` is a whole
/// number of quant blocks (`value_head_dim % group == 0` — always true for
/// Q1_0 g128 on Bonsai, where value_head_dim = 128 = one block) the
/// permutation moves whole blocks within each packed row and
/// [`reorder_packed_out_cols`] applies it byte-exactly. Callers must verify
/// the divisibility before taking the packed path.
pub fn packed_reorder_out_cols(hf_name: &str) -> bool {
    hf_name.ends_with(".linear_attn.out_proj.weight")
}

/// Byte-exact analog of [`reorder_out_cols`] on packed blocks: permute the
/// `value_head_dim/group`-block column groups of every packed row from GGUF
/// tiled order to HF grouped order. `raw` is `[n, k]` row-major packed.
pub fn reorder_packed_out_cols(
    raw: &[u8],
    hf_shape: &[usize],
    dims: &GdnDims,
    group: usize,
    block_bytes: usize,
) -> Result<Vec<u8>> {
    ensure!(
        hf_shape.len() == 2,
        "packed out_proj reorder expects 2-D [n,k], got {hf_shape:?}"
    );
    let (n, k) = (hf_shape[0], hf_shape[1]);
    ensure!(
        k == dims.num_v_heads * dims.value_head_dim,
        "packed out_proj reorder: k={k} != num_v_heads*value_head_dim {}",
        dims.num_v_heads * dims.value_head_dim
    );
    ensure!(
        dims.value_head_dim.is_multiple_of(group),
        "packed out_proj reorder: value_head_dim {} not a multiple of group {group}",
        dims.value_head_dim
    );
    let blocks_per_row = k / group;
    let blocks_per_head = dims.value_head_dim / group;
    let head_bytes = blocks_per_head * block_bytes;
    let row_bytes = blocks_per_row * block_bytes;
    ensure!(
        raw.len() == n * row_bytes,
        "packed out_proj reorder: raw len {} != n*row_bytes {}",
        raw.len(),
        n * row_bytes
    );
    let mut out = raw.to_vec();
    for row in 0..n {
        let ro = row * row_bytes;
        for hf in 0..dims.num_v_heads {
            let g = dims.gguf_head(hf);
            out[ro + hf * head_bytes..ro + (hf + 1) * head_bytes]
                .copy_from_slice(&raw[ro + g * head_bytes..ro + (g + 1) * head_bytes]);
        }
    }
    Ok(out)
}
