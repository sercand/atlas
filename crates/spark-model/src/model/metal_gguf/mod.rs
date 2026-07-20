// SPDX-License-Identifier: AGPL-3.0-only

//! Apple-Silicon serving model for Qwen3.5/3.6-family GGUF checkpoints
//! (Bonsai-27B and friends) — the metal-feature counterpart of the CUDA
//! `TransformerModel`.
//!
//! Architecture: the per-layer math lives in the vendor-agnostic
//! [`crate::forward::qwen3_5`] module (the same code the
//! `metal_qwen35_inference` example validated token-for-token against
//! `mlx_lm.generate`); this module owns what the example owned — weight
//! wiring from the [`WeightStore`], kernel-handle resolution, per-slot
//! KV/GDN state, scratch buffers, and the [`Model`] trait plumbing the
//! scheduler drives.
//!
//! Simplifications vs the CUDA path (deliberate, v1):
//! - Decode is single-token; `decode_batch` runs sequences serially
//!   (weight-bandwidth-bound, so serial decode is the honest shape).
//!   Prefill is token-BATCHED (`prefill.rs`: staged embeds → q1 GEMM
//!   tiles → chunked GDN/conv kernels) when the config allows, with a
//!   per-token fallback for planar weights / quantized KV.
//! - No speculative / MTP / LoRA / prefix-cache / paged-KV — those trait
//!   methods stub exactly like `NllbGpuModel`'s.
//! - KV is a contiguous per-slot cache (`LayerKvCache`), allocated at
//!   `alloc_sequence` and freed at `free_sequence`.
//!
//! Weight format: big projections + embedding + LM head arrive
//! keep-packed 1-bit ([`WeightDtype::PackedQ1_0`] via
//! `ATLAS_GGUF_NATIVE_Q1`, default-on for metal builds); small tensors
//! (norms, GDN a/b projections, conv1d, `A_log`, `dt_bias`) arrive BF16.
//! [`MetalQw`] dispatches per-tensor between the packed q1 kernels and
//! the dense BF16 GEMV so both kinds sit behind one [`QuantWeights`].

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::gguf_q1::{self, GgufQ1Weight};
use spark_runtime::weights::{WeightStore, WeightTensor};

use crate::forward::quant_weights::QuantWeights;
use crate::forward::qwen3_5::{
    FullAttentionScratch, LayerKvCache, LinearAttentionScratch, LinearAttentionState, MetalKvDtype,
    Qwen35ForwardConfig, Qwen35Kernels,
};
use crate::traits::Model;

mod forward;
mod init;
mod model_impl;
mod prefill;
mod vision;

/// A dense BF16 `[N, K]` weight driven through the `dense_gemv_bf16`
/// kernel — the fallback for the small tensors the GGUF loader does not
/// keep packed (GDN `in_proj_a`/`in_proj_b`, and every projection when
/// native-Q1 is disabled).
#[derive(Clone, Copy)]
pub struct DenseBf16Weight {
    pub ptr: DevicePtr,
    pub out_features: u32,
    pub in_features: u32,
}

impl DenseBf16Weight {
    fn gemv(&self, gpu: &dyn GpuBackend, x: DevicePtr, y: DevicePtr, stream: u64) -> Result<()> {
        let kernel = gpu.kernel("dense_gemv_bf16", "dense_gemv_bf16")?;
        gpu.launch_typed(
            kernel,
            [self.out_features, 1, 1],
            [256, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&self.out_features.to_le_bytes()),
                KernelArg::Bytes(&self.in_features.to_le_bytes()),
                KernelArg::Buffer(self.ptr),
                KernelArg::Buffer(x),
                KernelArg::Buffer(y),
            ],
        )
    }
}

/// Per-tensor quantization dispatch: keep-packed 1-bit or dense BF16.
/// One enum (rather than a generic parameter per kind) because a single
/// GDN layer mixes both — `in_proj_qkv` packed, `in_proj_a` dense.
#[derive(Clone, Copy)]
pub enum MetalQw {
    Q1(GgufQ1Weight),
    Dense(DenseBf16Weight),
}

impl MetalQw {
    /// Wrap a store tensor by its dtype tag.
    pub fn from_store(store: &WeightStore, name: &str) -> Result<Self> {
        let t = store.get(name)?;
        Self::from_tensor(t, name)
    }

    pub fn from_tensor(t: &WeightTensor, name: &str) -> Result<Self> {
        if t.is_packed_q1() {
            return Ok(Self::Q1(GgufQ1Weight::from_tensor(t, name)?));
        }
        if t.dtype == spark_runtime::weights::WeightDtype::BF16 {
            if t.shape.len() != 2 {
                bail!("{name}: expected 2-D BF16 weight, got {:?}", t.shape);
            }
            return Ok(Self::Dense(DenseBf16Weight {
                ptr: t.ptr,
                out_features: t.shape[0] as u32,
                in_features: t.shape[1] as u32,
            }));
        }
        bail!(
            "{name}: unsupported dtype {:?} for the metal GGUF model (expected PackedQ1_0 or BF16)",
            t.dtype
        );
    }
}

impl QuantWeights for MetalQw {
    fn out_features(&self) -> u32 {
        match self {
            Self::Q1(w) => w.out_features,
            Self::Dense(w) => w.out_features,
        }
    }
    fn in_features(&self) -> u32 {
        match self {
            Self::Q1(w) => w.in_features,
            Self::Dense(w) => w.in_features,
        }
    }
    fn gemv(&self, gpu: &dyn GpuBackend, x: DevicePtr, y: DevicePtr, stream: u64) -> Result<()> {
        match self {
            Self::Q1(w) => w.gemv(gpu, x, y, stream),
            Self::Dense(w) => w.gemv(gpu, x, y, stream),
        }
    }
    fn gemv_gate_up_with(
        &self,
        other: &Self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        gate_y: DevicePtr,
        up_y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        match (self, other) {
            // Both packed → fused dual-output kernel (one x read).
            (Self::Q1(a), Self::Q1(b)) => gguf_q1::gemv_gate_up(gpu, a, b, x, gate_y, up_y, stream),
            // Both dense with matching shapes → dual kernel (one
            // dispatch instead of two; the GDN in_proj_a/b pair).
            (Self::Dense(a), Self::Dense(b))
                if a.out_features == b.out_features && a.in_features == b.in_features =>
            {
                let kernel = gpu.kernel("dense_gemv_bf16", "dense_gemv_bf16_dual")?;
                gpu.launch_typed(
                    kernel,
                    [a.out_features, 1, 1],
                    [256, 1, 1],
                    0,
                    stream,
                    &[
                        KernelArg::Bytes(&a.out_features.to_le_bytes()),
                        KernelArg::Bytes(&a.in_features.to_le_bytes()),
                        KernelArg::Buffer(a.ptr),
                        KernelArg::Buffer(b.ptr),
                        KernelArg::Buffer(x),
                        KernelArg::Buffer(gate_y),
                        KernelArg::Buffer(up_y),
                    ],
                )
            }
            _ => {
                self.gemv(gpu, x, gate_y, stream)?;
                other.gemv(gpu, x, up_y, stream)
            }
        }
    }
    fn gemv_silu_gate(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Q1(w) => w.gemv_silu_gate(gpu, gate, up, y, stream),
            Self::Dense(_) => bail!(
                "gemv_silu_gate on a dense BF16 weight — the FFN down_proj should be keep-packed"
            ),
        }
    }
    fn gemv_silu_gate_resid(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        x_resid: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Q1(w) => w.gemv_silu_gate_resid(gpu, gate, up, x_resid, y, stream),
            Self::Dense(_) => bail!(
                "gemv_silu_gate_resid on a dense BF16 weight — the FFN down_proj should be keep-packed"
            ),
        }
    }
    fn gemv_ffn_swiglu(
        &self,
        gate_w: &Self,
        up_w: &Self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        gate_scratch: DevicePtr,
        act_scratch: DevicePtr,
        x_resid: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        // All-packed blocked layout → activation folded into the dual
        // gemv's epilogue (2 dispatches for the whole FFN tail).
        if let (Self::Q1(d), Self::Q1(g), Self::Q1(u)) = (self, gate_w, up_w)
            && !d.planar
            && !g.planar
            && !u.planar
        {
            return gguf_q1::gemv_ffn_swiglu(
                gpu, g, u, d, x, act_scratch, x_resid, y, stream,
            );
        }
        gate_w.gemv_gate_up_with(up_w, gpu, x, gate_scratch, act_scratch, stream)?;
        self.gemv_silu_gate_resid(gpu, gate_scratch, act_scratch, x_resid, y, stream)
    }
}

/// Full-attention layer weights (device pointers + quant projections).
pub(crate) struct MetalFullLayer {
    pub input_ln: DevicePtr,
    pub q_norm: DevicePtr,
    pub k_norm: DevicePtr,
    pub post_ln: DevicePtr,
    pub q_proj: MetalQw,
    pub k_proj: MetalQw,
    pub v_proj: MetalQw,
    pub o_proj: MetalQw,
    pub gate_proj: MetalQw,
    pub up_proj: MetalQw,
    pub down_proj: MetalQw,
}

/// Linear-attention (GDN) layer weights.
pub(crate) struct MetalLinLayer {
    pub input_ln: DevicePtr,
    /// FP32 `[num_v_heads]` (widened from the store's BF16 at init).
    pub a_log: DevicePtr,
    pub dt_bias: DevicePtr,
    pub conv1d: DevicePtr,
    pub norm_w: DevicePtr,
    pub post_ln: DevicePtr,
    pub in_proj_a: MetalQw,
    pub in_proj_b: MetalQw,
    pub in_proj_qkv: MetalQw,
    pub in_proj_z: MetalQw,
    pub out_proj: MetalQw,
    pub gate_proj: MetalQw,
    pub up_proj: MetalQw,
    pub down_proj: MetalQw,
}

pub(crate) enum MetalLayer {
    Full(MetalFullLayer),
    Linear(MetalLinLayer),
}

/// Per-sequence device state: contiguous KV for the 16 full-attention
/// layers + conv/GDN recurrent state for the 48 linear layers, plus the
/// request's encoded vision rows and MRoPE position plan.
pub(crate) struct SlotState {
    pub kv: Vec<LayerKvCache>,
    pub lin: Vec<LinearAttentionState>,
    /// Encoded image rows + splice cursor (taken from the model's
    /// pending slot on the first prefill chunk).
    pub vision: Option<vision::VisionRows>,
    /// Per-prompt-token (t, h, w) MRoPE positions, built on chunk 0.
    pub mrope: Vec<[u32; 3]>,
    /// Next decode position (continues past the image spatial extent).
    pub next_pos: u32,
}

/// Shared per-forward buffers (scratch, residual stream, rope tables).
/// One set, guarded by a mutex — forwards are serialized, which is the
/// throughput model of this backend anyway (weight-bandwidth-bound).
pub(crate) struct ForwardBufs {
    pub x_buf: DevicePtr,
    pub x_final: DevicePtr,
    pub positions: DevicePtr,
    pub inv_freq: DevicePtr,
    pub full_scratch: FullAttentionScratch,
    pub lin_scratch: LinearAttentionScratch,
    /// Host-side staging for the CPU embed-row dequant.
    pub embed_f32: Vec<f32>,
    pub embed_bf16: Vec<u8>,
    /// Prefill staging: `stage_cap` embedding rows (`[stage_cap,
    /// hidden]` BF16) + MRoPE triples (`[stage_cap, 3]` u32), uploaded
    /// once per prompt sub-chunk. The per-token prefill loop then reads
    /// its layer-0 input straight from a staged row, so it issues no
    /// host writes — and therefore needs no per-token synchronize.
    pub x_stage: DevicePtr,
    pub pos_stage: DevicePtr,
    pub stage_cap: usize,
    /// Host mirrors filled by the CPU embed dequant before the upload.
    pub stage_host: Vec<u8>,
    pub pos_host: Vec<u8>,
    /// Token-batched prefill scratch (`None` when the batched path is
    /// gated off — planar weights / non-BF16 KV / env kill-switch).
    pub prefill: Option<prefill::PrefillBufs>,
}

pub struct MetalGgufModel {
    pub(crate) gpu: Box<dyn GpuBackend>,
    pub(crate) cfg: Qwen35ForwardConfig,
    pub(crate) kernels: Qwen35Kernels,
    /// Batched-prefill kernel handles; `Some` iff the batched path is
    /// enabled (BF16 KV, blocked weights, 128×128 GDN heads).
    pub(crate) prefill_kernels: Option<prefill::PrefillKernels>,
    pub(crate) argmax: KernelHandle,
    pub(crate) layers: Vec<MetalLayer>,
    /// layer idx → ordinal among full-attention layers (KV slot index).
    pub(crate) kv_ord: Vec<Option<usize>>,
    /// layer idx → ordinal among linear-attention layers.
    pub(crate) lin_ord: Vec<Option<usize>>,
    pub(crate) final_norm: DevicePtr,
    pub(crate) lm_head: GgufQ1Weight,
    /// Host copy of the packed Q1 embedding table (`[vocab, hidden]`
    /// rows of 18-byte blocks) — embed lookup is a CPU row dequant.
    pub(crate) embed_host: Vec<u8>,
    pub(crate) max_seq_len: u32,
    pub(crate) max_batch: usize,
    pub(crate) kv_dtype: MetalKvDtype,
    pub(crate) fwd: Mutex<ForwardBufs>,
    pub(crate) free_slots: Mutex<Vec<usize>>,
    pub(crate) states: Mutex<HashMap<usize, SlotState>>,
    /// `[2 * max_batch, vocab]` BF16: rows `0..max_batch` are the decode
    /// batch rows (contiguous, batch position i ↔ row i), rows
    /// `max_batch..` are per-slot prefill rows.
    pub(crate) logits: DevicePtr,
    pub(crate) argmax_out: DevicePtr,
    /// Vision tower (present when the mmproj sidecar loaded).
    pub(crate) vision: Option<vision::MetalVision>,
    /// Rows encoded by `prepare_vision_embed`, awaiting the next
    /// first-chunk prefill (which moves them into that slot's state).
    pub(crate) pending_vision: Mutex<Option<vision::VisionRows>>,
}

impl MetalGgufModel {
    pub(crate) fn decode_logits_row(&self, i: usize) -> DevicePtr {
        self.logits.offset(i * self.cfg.vocab as usize * 2)
    }
    pub(crate) fn prefill_logits_row(&self, slot: usize) -> DevicePtr {
        self.decode_logits_row(self.max_batch + (slot % self.max_batch))
    }
}

/// True when this config is servable by the metal GGUF model: the
/// Qwen3.5/3.6 GDN-hybrid dense family (Bonsai-27B, Qwen3.5-4B, …).
pub fn supports(config: &ModelConfig) -> bool {
    let normalized = config.model_type.to_lowercase().replace(['-', '.'], "_");
    matches!(
        normalized.as_str(),
        "qwen3_5" | "qwen3_5_moe" | "qwen35" | "qwen35_moe" | "qwen3_6_moe"
    ) && config.num_experts == 0
}

/// Map the CLI KV dtype onto the metal contiguous-cache formats. FP8/NVFP4
/// have no metal kernels; they map to the TurboQuant format of the same
/// byte width (logged at build). Unknown asym combos fall back to the
/// safer-asym family or bf16.
fn metal_kv_dtype(kv: KvCacheDtype) -> MetalKvDtype {
    let mapped = match kv {
        KvCacheDtype::Bf16 => MetalKvDtype::Bf16,
        KvCacheDtype::Fp8 | KvCacheDtype::Turbo8 => MetalKvDtype::Turbo8,
        KvCacheDtype::Nvfp4 | KvCacheDtype::Turbo4 => MetalKvDtype::Turbo4,
        KvCacheDtype::Turbo3 => MetalKvDtype::Turbo3,
        KvCacheDtype::Turbo2 => MetalKvDtype::Turbo2,
        KvCacheDtype::Bf16KTurbo4V | KvCacheDtype::Fp8KTurbo4V | KvCacheDtype::Turbo4KTurbo8V => {
            MetalKvDtype::Bf16KTurbo4V
        }
        KvCacheDtype::Bf16KTurbo3V
        | KvCacheDtype::Fp8KTurbo3V
        | KvCacheDtype::Turbo4KTurbo3V
        | KvCacheDtype::Turbo3KTurbo8V => MetalKvDtype::Bf16KTurbo3V,
        KvCacheDtype::Bf16KTurbo2V | KvCacheDtype::Fp8KTurbo2V => MetalKvDtype::Bf16KTurbo2V,
    };
    if format!("{kv:?}") != format!("{mapped:?}") {
        tracing::info!("metal KV cache: --kv-cache-dtype {kv:?} mapped to {mapped:?}");
    }
    mapped
}

/// Build the metal serving model. Entry point for `factory::build_model`
/// under the metal feature.
pub fn build_metal_model(
    config: &ModelConfig,
    store: &WeightStore,
    gpu: Box<dyn GpuBackend>,
    max_seq_len: usize,
    max_batch_size: usize,
    kv_dtype: KvCacheDtype,
) -> Result<Box<dyn Model>> {
    if !supports(config) {
        bail!(
            "Unsupported model type: '{}' on the metal backend \
             (supported: dense Qwen3.5/3.6 GDN-hybrid family, e.g. Bonsai-27B)",
            config.model_type
        );
    }
    let model = init::build(config, store, gpu, max_seq_len, max_batch_size, kv_dtype)
        .context("building MetalGgufModel")?;
    Ok(Box::new(model))
}
