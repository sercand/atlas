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
//! loader (`WeightDtype::PackedQ1_0`, or `PackedQ1Planar` under
//! `ATLAS_Q1_PLANAR=1`) and drives the Metal kernels: `q1_0_gemv`
//! (+`_resid`, `_batchm`), `q1_0_gemv_gate_up` (+`_act`), and the
//! elementwise `silu_gate` for the composed FFN tail. On Apple-Silicon
//! UMA the same packed bytes are also readable host-side, which
//! [`dequant_row_f32`] uses for CPU embedding-row lookups.

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
    /// `[out_features, in_features/128]` packed 18-byte blocks — in
    /// on-disk block order, or the loader's row-planar reorder when
    /// `planar` (see `WeightDtype::PackedQ1Planar`).
    pub packed: DevicePtr,
    pub out_features: u32,
    pub in_features: u32,
    /// Row-planar byte order → dispatch the `_planar` kernel variants.
    pub planar: bool,
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
            planar: t.dtype == crate::weights::WeightDtype::PackedQ1Planar,
        })
    }

    /// Kernel function name honoring the tensor's byte order.
    fn fn_name(&self, base: &'static str, planar_variant: &'static str) -> &'static str {
        if self.planar { planar_variant } else { base }
    }

    /// Row-group launch geometry shared by the q1 gemv kernels: 16
    /// rows per 128-thread threadgroup (4 simdgroups × 4 rows each).
    /// 4 rows/simdgroup measured fastest — 8 amortizes x further but
    /// the doubled accumulator/stream state costs more than it saves.
    fn grid(&self) -> ([u32; 3], [u32; 3]) {
        ([self.out_features.div_ceil(16), 1, 1], [128, 1, 1])
    }

    /// Launch geometry for the dual-stream gate_up kernels.
    fn grid_gate_up(&self) -> ([u32; 3], [u32; 3]) {
        ([self.out_features.div_ceil(16), 1, 1], [128, 1, 1])
    }

    /// Legacy geometry for `q1_0_gemv_batchm` (one simdgroup per row).
    fn grid_batchm(&self) -> ([u32; 3], [u32; 3]) {
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
        let kernel = gpu.kernel("q1_0_gemv", self.fn_name("q1_0_gemv", "q1_0_gemv_planar"))?;
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
        if self.planar {
            bail!("q1_0_gemv_batchm has no planar variant (decode path is single-token)");
        }
        let kernel = gpu.kernel("q1_0_gemv", "q1_0_gemv_batchm")?;
        let (grid, block) = self.grid_batchm();
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

    /// Elementwise `up[k] = silu(gate[k]) ⊙ up[k]` over `in_features`
    /// elements. The in-place clobber of `up` is deliberate: the
    /// silu-gate entry points receive per-layer scratch that this call
    /// consumes, and writing the activation once here is what removes
    /// the N-fold silu recompute the old fused gemv kernels paid
    /// (activation cost scaled with OUTPUT rows, not input length).
    fn activate_silu(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let kernel = gpu.kernel("silu_gate", "silu_gate")?;
        gpu.launch_typed(
            kernel,
            [self.in_features.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&self.in_features.to_le_bytes()),
                KernelArg::Buffer(gate),
                KernelArg::Buffer(up),
                KernelArg::Buffer(up),
            ],
        )
    }

    /// FFN tail: `y = W @ (silu(gate) ⊙ up)`. Clobbers `up` with the
    /// activated vector (see [`Self::activate_silu`]).
    pub fn gemv_silu_gate(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        self.activate_silu(gpu, gate, up, stream)?;
        self.gemv(gpu, up, y, stream)
    }

    /// Matvec with the residual-stream add folded into the epilogue:
    /// `y[n] = x_resid[n] + Σ_k W[n,k]·x[k]`.
    pub fn gemv_resid(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        x_resid: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let kernel = gpu.kernel(
            "q1_0_gemv",
            self.fn_name("q1_0_gemv_resid", "q1_0_gemv_resid_planar"),
        )?;
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
                KernelArg::Buffer(x_resid),
                KernelArg::Buffer(y),
            ],
        )
    }

    /// Like [`Self::gemv_silu_gate`] with the residual add folded into
    /// the gemv: `y[n] = x_resid[n] + Σ_k W[n,k]·(silu(gate)⊙up)[k]`.
    /// Clobbers `up`.
    pub fn gemv_silu_gate_resid(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        x_resid: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        self.activate_silu(gpu, gate, up, stream)?;
        self.gemv_resid(gpu, up, x_resid, y, stream)
    }
}

/// Whole SwiGLU FFN tail in two dispatches:
/// `y = x_resid + down_w @ (silu(gate_w @ x) ⊙ (up_w @ x))`.
/// The `q1_0_gemv_gate_up_act` kernel folds the activation into the
/// dual-gemv epilogue (writing `act_scratch`), so no elementwise silu
/// dispatch runs between the two gemvs. Blocked layout only — callers
/// fall back to the composed path for planar tensors.
pub fn gemv_ffn_swiglu(
    gpu: &dyn GpuBackend,
    gate_w: &GgufQ1Weight,
    up_w: &GgufQ1Weight,
    down_w: &GgufQ1Weight,
    x: DevicePtr,
    act_scratch: DevicePtr,
    x_resid: DevicePtr,
    y: DevicePtr,
    stream: u64,
) -> Result<()> {
    if gate_w.out_features != up_w.out_features || gate_w.in_features != up_w.in_features {
        bail!(
            "ffn gate/up shape mismatch: [{}, {}] vs [{}, {}]",
            gate_w.out_features,
            gate_w.in_features,
            up_w.out_features,
            up_w.in_features
        );
    }
    if gate_w.planar || up_w.planar || down_w.planar {
        bail!("gemv_ffn_swiglu: planar tensors take the composed path");
    }
    let kernel = gpu.kernel("q1_0_gemv_gate_up", "q1_0_gemv_gate_up_act")?;
    let (grid, block) = gate_w.grid_gate_up();
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
            KernelArg::Buffer(act_scratch),
        ],
    )?;
    down_w.gemv_resid(gpu, act_scratch, x_resid, y, stream)
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
    if gate_w.planar != up_w.planar {
        bail!("gate/up byte-order mismatch (one planar, one blocked)");
    }
    let kernel = gpu.kernel(
        "q1_0_gemv_gate_up",
        gate_w.fn_name("q1_0_gemv_gate_up", "q1_0_gemv_gate_up_planar"),
    )?;
    let (grid, block) = gate_w.grid_gate_up();
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
