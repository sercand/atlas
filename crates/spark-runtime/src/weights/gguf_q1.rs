// SPDX-License-Identifier: AGPL-3.0-only

//! Keep-packed PrismML Q1_0 (ggml id 41) weight support — the 1-bit
//! binary format the Bonsai-27B GGUF checkpoints ship.
//!
//! Block layout (fixed group 128, 18 bytes, canonical here and mirrored
//! by `q1_0_gemv.metal` and the GGUF loader's CPU oracle):
//!
//!   [ fp16 d (little-endian) ][ 16 bytes of sign bits ]
//!   value(j) = bit(qs[j/8] >> (j%8)) ? +d : -d      (LSB-first)
//!
//! Row `n` of an `[N, K]` weight = `K/128` contiguous blocks starting at
//! byte offset `n * (K/128) * 18`; `K % 128 == 0` always holds (every
//! Bonsai linear and the embedding/LM-head rows are multiples of 128).
//!
//! [`GgufQ1Weight`] wraps a packed device buffer produced by the GGUF
//! loader (`WeightDtype::PackedQ1_0`) and drives the fused Metal
//! kernels: `q1_0_gemv` (+`_batchm`), `q1_0_gemv_gate_up`, and
//! `q1_0_gemv_silu_gate(_resid)`. On Apple-Silicon UMA the same packed
//! bytes are also readable host-side, which [`dequant_row_f32`] uses for
//! CPU embedding-row lookups.

use anyhow::{Result, bail};

use crate::gpu::{DevicePtr, GpuBackend, KernelArg};
use crate::weights::{WeightStore, WeightTensor};

/// Elements per Q1_0 block (`QK1_0` in the PrismML llama.cpp fork).
pub const Q1_GROUP: usize = 128;
/// On-disk / in-memory bytes per block: fp16 scale + 128 sign bits.
pub const Q1_BLOCK_BYTES: usize = 18;

/// Dequantize one 18-byte Q1_0 block into 128 f32 values. This is the
/// canonical CPU reference for the format — the GGUF loader's oracle
/// delegates here so the loader, the embed lookup, and the kernel
/// parity tests can never disagree on the bit layout.
pub fn dequant_block_f32(blk: &[u8], out: &mut [f32]) {
    let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
    let qs = &blk[2..2 + 16];
    for j in 0..Q1_GROUP {
        let bit = (qs[j / 8] >> (j % 8)) & 1;
        out[j] = if bit != 0 { d } else { -d };
    }
}

/// Dequantize one row of a packed `[N, K]` Q1_0 weight (`row_bytes` =
/// the row's `K/128` contiguous blocks) into f32. Used for CPU-side
/// embedding lookups on UMA.
pub fn dequant_row_f32(row_bytes: &[u8], k: usize, out: &mut [f32]) -> Result<()> {
    if !k.is_multiple_of(Q1_GROUP) {
        bail!("Q1_0 row length {k} not a multiple of {Q1_GROUP}");
    }
    let blocks = k / Q1_GROUP;
    if row_bytes.len() < blocks * Q1_BLOCK_BYTES {
        bail!(
            "Q1_0 row bytes too short: {} < {}",
            row_bytes.len(),
            blocks * Q1_BLOCK_BYTES
        );
    }
    for b in 0..blocks {
        dequant_block_f32(
            &row_bytes[b * Q1_BLOCK_BYTES..(b + 1) * Q1_BLOCK_BYTES],
            &mut out[b * Q1_GROUP..(b + 1) * Q1_GROUP],
        );
    }
    Ok(())
}

/// One keep-packed Q1_0 linear weight resident on the GPU. The pointer
/// is BORROWED from the [`WeightStore`] (freed at store teardown, not
/// here), mirroring how the CUDA `PackedQ2Weight` wraps store buffers.
#[derive(Clone, Copy)]
pub struct GgufQ1Weight {
    /// `[out_features, in_features/128]` packed 18-byte blocks.
    pub packed: DevicePtr,
    pub out_features: u32,
    pub in_features: u32,
}

impl GgufQ1Weight {
    /// Wrap a `WeightDtype::PackedQ1_0` store entry, validating shape.
    pub fn from_store(store: &WeightStore, name: &str) -> Result<Self> {
        Self::from_tensor(store.get(name)?, name)
    }

    /// Wrap an already-fetched store tensor.
    pub fn from_tensor(t: &WeightTensor, name: &str) -> Result<Self> {
        if !t.is_packed_q1() {
            bail!("{name}: expected PackedQ1_0, got {:?}", t.dtype);
        }
        if t.shape.len() != 2 {
            bail!("{name}: expected 2-D packed weight, got {:?}", t.shape);
        }
        let (n, k) = (t.shape[0], t.shape[1]);
        if !k.is_multiple_of(Q1_GROUP) {
            bail!("{name}: K {k} not a multiple of {Q1_GROUP}");
        }
        Ok(Self {
            packed: t.ptr,
            out_features: n as u32,
            in_features: k as u32,
        })
    }

    /// Row-group launch geometry shared by every q1 gemv kernel:
    /// 4 rows per threadgroup, one 32-lane simdgroup per row.
    fn grid(&self) -> ([u32; 3], [u32; 3]) {
        ([self.out_features.div_ceil(4), 1, 1], [128, 1, 1])
    }

    /// Decode-path matvec `y = W @ x` (x BF16 `[K]`, y BF16 `[N]`).
    pub fn gemv(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let kernel = gpu.kernel("q1_0_gemv", "q1_0_gemv")?;
        let (grid, block) = self.grid();
        gpu.launch_typed(
            kernel,
            grid,
            block,
            0,
            stream,
            &[
                KernelArg::Bytes(&self.out_features.to_le_bytes()),
                KernelArg::Bytes(&self.in_features.to_le_bytes()),
                KernelArg::Buffer(self.packed),
                KernelArg::Buffer(x),
                KernelArg::Buffer(y),
            ],
        )
    }

    /// Batched decode matvec: `y[m] = W @ x[m]` for `m` co-scheduled
    /// lanes (x `[M, K]`, y `[M, N]`). Weights are read once per row
    /// regardless of `M`.
    pub fn gemv_batchm(
        &self,
        gpu: &dyn GpuBackend,
        m: u32,
        x: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if m > 8 {
            bail!("q1_0_gemv_batchm supports M <= 8, got {m}");
        }
        let kernel = gpu.kernel("q1_0_gemv", "q1_0_gemv_batchm")?;
        let (grid, block) = self.grid();
        gpu.launch_typed(
            kernel,
            grid,
            block,
            0,
            stream,
            &[
                KernelArg::Bytes(&self.out_features.to_le_bytes()),
                KernelArg::Bytes(&self.in_features.to_le_bytes()),
                KernelArg::Bytes(&m.to_le_bytes()),
                KernelArg::Buffer(self.packed),
                KernelArg::Buffer(x),
                KernelArg::Buffer(y),
            ],
        )
    }

    /// Fused FFN tail: `y = W @ (silu(gate) ⊙ up)`.
    pub fn gemv_silu_gate(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let kernel = gpu.kernel("q1_0_gemv_silu_gate", "q1_0_gemv_silu_gate")?;
        let (grid, block) = self.grid();
        gpu.launch_typed(
            kernel,
            grid,
            block,
            0,
            stream,
            &[
                KernelArg::Bytes(&self.out_features.to_le_bytes()),
                KernelArg::Bytes(&self.in_features.to_le_bytes()),
                KernelArg::Buffer(self.packed),
                KernelArg::Buffer(gate),
                KernelArg::Buffer(up),
                KernelArg::Buffer(y),
            ],
        )
    }

    /// Like [`Self::gemv_silu_gate`] with the residual add folded in:
    /// `y[n] = x_resid[n] + Σ_k W[n,k]·(silu(gate)⊙up)[k]`.
    pub fn gemv_silu_gate_resid(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        x_resid: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let kernel = gpu.kernel("q1_0_gemv_silu_gate", "q1_0_gemv_silu_gate_resid")?;
        let (grid, block) = self.grid();
        gpu.launch_typed(
            kernel,
            grid,
            block,
            0,
            stream,
            &[
                KernelArg::Bytes(&self.out_features.to_le_bytes()),
                KernelArg::Bytes(&self.in_features.to_le_bytes()),
                KernelArg::Buffer(self.packed),
                KernelArg::Buffer(gate),
                KernelArg::Buffer(up),
                KernelArg::Buffer(x_resid),
                KernelArg::Buffer(y),
            ],
        )
    }
}

/// Fused dual-output GEMV: `gate_y = gate_w @ x`, `up_y = up_w @ x`,
/// sharing one read of `x`. Free function (not a method) because it
/// spans two weights, mirroring `mlx_int8::gemv_gate_up`.
pub fn gemv_gate_up(
    gpu: &dyn GpuBackend,
    gate_w: &GgufQ1Weight,
    up_w: &GgufQ1Weight,
    x: DevicePtr,
    gate_y: DevicePtr,
    up_y: DevicePtr,
    stream: u64,
) -> Result<()> {
    if gate_w.out_features != up_w.out_features || gate_w.in_features != up_w.in_features {
        bail!(
            "gate/up shape mismatch: [{}, {}] vs [{}, {}]",
            gate_w.out_features,
            gate_w.in_features,
            up_w.out_features,
            up_w.in_features
        );
    }
    let kernel = gpu.kernel("q1_0_gemv_gate_up", "q1_0_gemv_gate_up")?;
    let (grid, block) = gate_w.grid();
    gpu.launch_typed(
        kernel,
        grid,
        block,
        0,
        stream,
        &[
            KernelArg::Bytes(&gate_w.out_features.to_le_bytes()),
            KernelArg::Bytes(&gate_w.in_features.to_le_bytes()),
            KernelArg::Buffer(gate_w.packed),
            KernelArg::Buffer(up_w.packed),
            KernelArg::Buffer(x),
            KernelArg::Buffer(gate_y),
            KernelArg::Buffer(up_y),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_dequant_matches_bit_semantics() {
        // d = 1.5; bits 0b0000_0101 -> +d, -d, +d, -d, ...
        let mut blk = vec![0u8; Q1_BLOCK_BYTES];
        let d = half::f16::from_f32(1.5);
        blk[..2].copy_from_slice(&d.to_le_bytes());
        blk[2] = 0x05;
        let mut out = [0f32; Q1_GROUP];
        dequant_block_f32(&blk, &mut out);
        assert_eq!(out[0], 1.5);
        assert_eq!(out[1], -1.5);
        assert_eq!(out[2], 1.5);
        assert_eq!(out[3], -1.5);
        assert_eq!(out[127], -1.5);
    }

    #[test]
    fn row_dequant_walks_blocks() {
        let k = 256;
        let mut row = Vec::new();
        for b in 0..2 {
            let d = half::f16::from_f32(1.0 + b as f32);
            row.extend_from_slice(&d.to_le_bytes());
            row.extend_from_slice(&[0xFFu8; 16]); // all +d
        }
        let mut out = vec![0f32; k];
        dequant_row_f32(&row, k, &mut out).unwrap();
        assert_eq!(out[0], 1.0);
        assert_eq!(out[127], 1.0);
        assert_eq!(out[128], 2.0);
        assert_eq!(out[255], 2.0);
    }

    #[test]
    fn rejects_bad_row_geometry() {
        let mut out = vec![0f32; 100];
        assert!(dequant_row_f32(&[0u8; 18], 100, &mut out).is_err());
        let mut out = vec![0f32; 128];
        assert!(dequant_row_f32(&[0u8; 10], 128, &mut out).is_err());
    }
}
