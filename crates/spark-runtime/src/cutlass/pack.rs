// SPDX-License-Identifier: AGPL-3.0-only
//! CUTLASS NVFP4 weight pack / scale-swizzle / transpose host wrappers.

use anyhow::{Result, bail};

#[cfg(atlas_cutlass)]
use std::ffi::c_void;

#[cfg(atlas_cutlass)]
use super::*;

/// Repack an Atlas E4M3 weight scale into the CUTLASS SM120 blockscaled SFB
/// swizzle atom (`tile_atom_to_shape_SFB`, ue4m3) that the grouped collective
/// reads. M-independent (the SFB atom depends only on N,K) so this runs once
/// per expert at load. `scale_out` must hold the swizzled SFB region the
/// grouped kernel consumes.
///
/// `src_n_major` selects the SOURCE layout: `false` = Atlas-transposed
/// `[K/16,N]`, `true` = checkpoint-native `[N,K/16]`. The N-major mode lets a
/// checkpoint that already ships `[N,K/16]` scales (Laguna) build SFB without
/// first materialising an Atlas-transposed copy. Output layout is identical
/// either way.
pub fn pack_weight_sfb(
    scale_in: u64,
    scale_out: u64,
    n: u32,
    k: u32,
    src_n_major: bool,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let status = unsafe {
            atlas_cutlass_pack_weight_sfb(
                scale_in as *const c_void,
                scale_out as *mut c_void,
                n as i32,
                k as i32,
                i32::from(src_n_major),
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS weight SFB pack failed: status {status} for {n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (scale_in, scale_out, n, k, src_n_major, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// [`pack_weight_sfb`] for every expert of one projection in a single launch.
///
/// `in_ptrs`/`out_ptrs` are DEVICE arrays of `num_experts` device pointers (a
/// null entry on either side skips that expert); `n`/`k`/`src_n_major` are the
/// same for all experts, which is what makes one launch legal. Output bytes are
/// identical to the per-expert loop.
///
/// This exists so the SFB tables can be re-derived per prefill call instead of
/// living resident: 512 experts x 3 projections x 48 layers is 7.6 GB of scale
/// swizzle that otherwise comes straight out of the KV pool. The per-expert
/// entry is kept for the resident (load-time) path and as the equivalence
/// reference.
pub fn pack_weight_sfb_batched(
    in_ptrs: u64,
    out_ptrs: u64,
    num_experts: u32,
    n: u32,
    k: u32,
    src_n_major: bool,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let status = unsafe {
            atlas_cutlass_pack_weight_sfb_batched(
                in_ptrs as *const c_void,
                out_ptrs as *const c_void,
                num_experts as i32,
                n as i32,
                k as i32,
                i32::from(src_n_major),
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!(
                "CUTLASS batched weight SFB pack failed: status {status} for \
                 {num_experts}x{n}x{k}"
            );
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (in_ptrs, out_ptrs, num_experts, n, k, src_n_major, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// Pack BF16 row-major weight `[N,K]` into the native CUTLASS NVFP4 layout:
/// packed `[N,K/2]` (N-major, K-contiguous — NOT the Atlas transposed `[K/2,N]`)
/// and E4M3 scales `[K/16,N]`. `weight_scale_2` is assumed to be 1.0 by the
/// caller when feeding this into the native CUTLASS wrapper.
pub fn pack_bf16_weight_to_nvfp4_t(
    weight_bf16: u64,
    packed_t: u64,
    scale_t: u64,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let status = unsafe {
            atlas_cutlass_pack_bf16_weight_to_nvfp4_t(
                weight_bf16 as *const c_void,
                packed_t as *mut c_void,
                scale_t as *mut c_void,
                n as i32,
                k as i32,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS BF16->NVFP4 weight pack failed: status {status} for {n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (weight_bf16, packed_t, scale_t, n, k, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// Transpose an Atlas-packed NVFP4 weight from the checkpoint/hand-kernel
/// `[K/2, N]` layout into CUTLASS's `[N, K/2]` layout (the byte order the
/// native NVFP4 GEMM consumes for the ColumnMajor B operand). Pure byte
/// transpose; nibble pairing within each byte is preserved. `dst_packed` must
/// have `N * K/2` bytes.
pub fn transpose_nvfp4_packed_kton(
    src_packed_t: u64,
    dst_packed: u64,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let status = unsafe {
            atlas_cutlass_transpose_nvfp4_packed_kton(
                src_packed_t as *const c_void,
                dst_packed as *mut c_void,
                n as i32,
                k as i32,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS NVFP4 weight transpose failed: status {status} for {n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (src_packed_t, dst_packed, n, k, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}
