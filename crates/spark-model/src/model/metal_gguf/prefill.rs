// SPDX-License-Identifier: AGPL-3.0-only

//! Batched (multi-token) prefill for the metal GGUF model.
//!
//! `run_tile_batched` pushes a TILE of up to [`PREFILL_TILE`] staged
//! prompt tokens through one decoder layer at a time with token-batched
//! kernels: the packed-Q1 projections go through `q1_0_gemm` (weights
//! stream once per 8 tokens instead of once per token), attention uses
//! `attention_prefill_offset` over the KV cache, and the GDN core runs
//! `gated_delta_rule_prefill` — ONE dispatch per head per tile with the
//! state held in registers, versus a full 6.3 MB state read+write per
//! token on the decode path.
//!
//! Gating (checked at init, `MetalGgufModel::prefill`): BF16 KV cache,
//! blocked (non-planar) Q1 weights, 128×128 GDN heads, and
//! `ATLAS_METAL_PREFILL_BATCH` unset or != 0. Anything else falls back
//! to the per-token loop in `model_impl.rs`.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};

use super::{ForwardBufs, MetalFullLayer, MetalGgufModel, MetalLayer, MetalLinLayer, MetalQw, SlotState};

/// Tokens per batched tile. Weight amortization saturates at the
/// kernel's 8-token inner tile; the OUTER tile mostly amortizes the
/// per-tile GDN state stream (48 layers × 6.3 MB per tile) and the
/// dispatch count, so bigger is better until scratch (~55 MB at 256)
/// starts to matter on a 16 GB machine.
pub(super) const PREFILL_TILE: usize = 256;

/// Pre-resolved handles for the prefill-only kernels.
pub(crate) struct PrefillKernels {
    attn: KernelHandle,
    kvap: KernelHandle,
    qkv_split: KernelHandle,
    gate_beta: KernelHandle,
    conv: KernelHandle,
    conv_state: KernelHandle,
    gdn: KernelHandle,
    silu: KernelHandle,
    sigmoid: KernelHandle,
    cast_half: KernelHandle,
}

impl PrefillKernels {
    /// `gdn_f16` must match the model's GDN-state dtype — the prefill
    /// and decode kernels share the per-slot state buffers.
    pub(super) fn resolve(gpu: &dyn GpuBackend, gdn_f16: bool) -> Result<Self> {
        Ok(Self {
            attn: gpu.kernel("attention_prefill", "attention_prefill_offset")?,
            kvap: gpu.kernel("kv_cache_append", "kv_cache_append_batch")?,
            qkv_split: gpu.kernel("qwen35_qkv_split", "qwen35_qkv_split_batch")?,
            gate_beta: gpu.kernel("gdn_helpers", "gdn_gate_beta_batch")?,
            conv: gpu.kernel("causal_conv1d_prefill", "causal_conv1d_prefill_l2norm")?,
            conv_state: gpu.kernel("causal_conv1d_prefill", "causal_conv1d_prefill_state")?,
            gdn: gpu.kernel(
                "gated_delta_rule_prefill",
                if gdn_f16 {
                    "gated_delta_rule_prefill_f16"
                } else {
                    "gated_delta_rule_prefill"
                },
            )?,
            silu: gpu.kernel("silu_gate", "silu_gate")?,
            sigmoid: gpu.kernel("sigmoid_gate", "sigmoid_gate")?,
            cast_half: gpu.kernel("gdn_helpers", "bf16_to_half_rows")?,
        })
    }
}

/// Token-batched scratch (`[PREFILL_TILE, …]` device mats). Lives in
/// `ForwardBufs` behind the forward mutex like the per-token scratch.
pub(crate) struct PrefillBufs {
    pub tile: usize,
    x_norm: DevicePtr,
    q_full: DevicePtr,
    q: DevicePtr,
    gate: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    attn: DevicePtr,
    o: DevicePtr,
    x_resid: DevicePtr,
    x_norm2: DevicePtr,
    gate_act: DevicePtr,
    up_act: DevicePtr,
    dt: DevicePtr,
    b: DevicePtr,
    qkv: DevicePtr,
    qkv_smooth: DevicePtr,
    z: DevicePtr,
    gatef: DevicePtr,
    betaf: DevicePtr,
    y: DevicePtr,
    y_norm: DevicePtr,
    /// Half-precision cast of the current packed-GEMM input, `[tile
    /// rounded to 128, max K]` — the GEMM's A fragments simdgroup_load
    /// straight from it (see `bf16_to_half_rows`).
    x_half: DevicePtr,
}

impl PrefillBufs {
    pub(super) fn alloc(
        gpu: &dyn GpuBackend,
        cfg: &crate::forward::qwen3_5::Qwen35ForwardConfig,
        tile: usize,
    ) -> Result<Self> {
        let bf16 = |n: u32| -> Result<DevicePtr> { gpu.alloc(tile * n as usize * 2) };
        let f32b = |n: u32| -> Result<DevicePtr> { gpu.alloc(tile * n as usize * 4) };
        Ok(Self {
            tile,
            x_norm: bf16(cfg.hidden)?,
            q_full: bf16(cfg.q_total())?,
            q: bf16(cfg.q_only())?,
            gate: bf16(cfg.q_only())?,
            k: bf16(cfg.kv_dim())?,
            v: bf16(cfg.kv_dim())?,
            attn: bf16(cfg.q_only())?,
            o: bf16(cfg.hidden)?,
            x_resid: bf16(cfg.hidden)?,
            x_norm2: bf16(cfg.hidden)?,
            gate_act: bf16(cfg.intermediate)?,
            up_act: bf16(cfg.intermediate)?,
            dt: bf16(cfg.num_state_heads())?,
            b: bf16(cfg.num_state_heads())?,
            qkv: bf16(cfg.qkv_total_lin())?,
            qkv_smooth: bf16(cfg.qkv_total_lin())?,
            z: bf16(cfg.z_dim_lin())?,
            gatef: f32b(cfg.num_state_heads())?,
            betaf: f32b(cfg.num_state_heads())?,
            y: bf16(cfg.z_dim_lin())?,
            y_norm: bf16(cfg.z_dim_lin())?,
            x_half: gpu.alloc(
                tile.next_multiple_of(128)
                    * cfg.hidden.max(cfg.intermediate).max(cfg.q_only()) as usize
                    * 2,
            )?,
        })
    }
}

impl MetalQw {
    /// Prefill-path matmul over `t` staged token rows (`y` = `[t, N]`).
    /// The packed path reads the pre-cast HALF input (`x_half`, from
    /// `bf16_to_half_rows`); the dense path reads the original BF16
    /// rows (`x_bf16`).
    fn gemm(
        &self,
        gpu: &dyn GpuBackend,
        t: u32,
        x_bf16: DevicePtr,
        x_half: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Q1(w) => w.gemm(gpu, t, x_half, y, stream),
            Self::Dense(w) => {
                let kernel = gpu.kernel("dense_gemm_bf16", "dense_gemm_bf16")?;
                gpu.launch_typed(
                    kernel,
                    [w.out_features.div_ceil(64), t, 1],
                    [64, 1, 1],
                    0,
                    stream,
                    &[
                        KernelArg::Bytes(&t.to_le_bytes()),
                        KernelArg::Bytes(&w.out_features.to_le_bytes()),
                        KernelArg::Bytes(&w.in_features.to_le_bytes()),
                        KernelArg::Buffer(x_bf16),
                        KernelArg::Buffer(w.ptr),
                        KernelArg::Buffer(y),
                    ],
                )
            }
        }
    }

    fn gemm_resid(
        &self,
        gpu: &dyn GpuBackend,
        t: u32,
        x_half: DevicePtr,
        x_resid: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Q1(w) => w.gemm_resid(gpu, t, x_half, x_resid, y, stream),
            Self::Dense(_) => bail!("gemm_resid on a dense BF16 weight (down_proj is packed)"),
        }
    }
}

impl MetalGgufModel {
    /// Run `n` staged tokens (rows of `x_mat` / triples at `pos_ptr`)
    /// through every decoder layer, batched. `base_pos` is the absolute
    /// KV position of row 0. Leaves each token's final residual in its
    /// `x_mat` row.
    pub(super) fn run_tile_batched(
        &self,
        bufs: &ForwardBufs,
        st: &SlotState,
        x_mat: DevicePtr,
        pos_ptr: DevicePtr,
        base_pos: u32,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        let pk = self
            .prefill_kernels
            .as_ref()
            .context("batched prefill kernels not resolved")?;
        let pb = bufs
            .prefill
            .as_ref()
            .context("batched prefill scratch not allocated")?;
        debug_assert!(n as usize <= pb.tile);
        for (idx, layer) in self.layers.iter().enumerate() {
            if idx > 0 && idx.is_multiple_of(16) {
                self.gpu.flush(stream)?;
            }
            match layer {
                MetalLayer::Full(l) => {
                    let kv = &st.kv[self.kv_ord[idx].expect("kv ordinal for full layer")];
                    self.full_layer_batched(pk, pb, l, kv, bufs, x_mat, pos_ptr, base_pos, n, stream)
                        .with_context(|| format!("layer {idx} (full attention, batched)"))?;
                }
                MetalLayer::Linear(l) => {
                    let state = &st.lin[self.lin_ord[idx].expect("lin ordinal for GDN layer")];
                    self.lin_layer_batched(pk, pb, l, state, x_mat, n, stream)
                        .with_context(|| format!("layer {idx} (GDN, batched)"))?;
                }
            }
        }
        Ok(())
    }

    /// Cast `n` rows of `[k]` BF16 at `x` into the shared half scratch
    /// (`pb.x_half`), zero-filling the token-tile padding rows the
    /// GEMM's simdgroup loads may touch. Returns the scratch pointer.
    fn cast_half(
        &self,
        pk: &PrefillKernels,
        pb: &PrefillBufs,
        x: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<DevicePtr> {
        let n_valid = n * k;
        let n_total = n.next_multiple_of(128) * k;
        self.gpu.launch_typed(
            pk.cast_half,
            [n_total.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&n_valid.to_le_bytes()),
                KernelArg::Bytes(&n_total.to_le_bytes()),
                KernelArg::Buffer(x),
                KernelArg::Buffer(pb.x_half),
            ],
        )?;
        Ok(pb.x_half)
    }

    /// One threadgroup-per-row RMSNorm over `rows` rows of `width`.
    fn rms_rows(
        &self,
        rows: u32,
        width: u32,
        block: u32,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        self.gpu.launch_typed(
            self.kernels.rms,
            [rows, 1, 1],
            [block, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&width.to_le_bytes()),
                KernelArg::Bytes(&self.cfg.rms_eps.to_le_bytes()),
                KernelArg::Buffer(x),
                KernelArg::Buffer(w),
                KernelArg::Buffer(out),
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn full_layer_batched(
        &self,
        pk: &PrefillKernels,
        pb: &PrefillBufs,
        l: &MetalFullLayer,
        kv: &crate::forward::qwen3_5::LayerKvCache,
        bufs: &ForwardBufs,
        x_mat: DevicePtr,
        pos_ptr: DevicePtr,
        base_pos: u32,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let gpu = self.gpu.as_ref();
        self.rms_rows(n, cfg.hidden, 512, x_mat, l.input_ln, pb.x_norm, stream)?;
        let xh = self.cast_half(pk, pb, pb.x_norm, n, cfg.hidden, stream)?;
        l.q_proj.gemm(gpu, n, pb.x_norm, xh, pb.q_full, stream)?;
        l.k_proj.gemm(gpu, n, pb.x_norm, xh, pb.k, stream)?;
        l.v_proj.gemm(gpu, n, pb.x_norm, xh, pb.v, stream)?;
        // Deinterleave [Q_h | gate_h] → q / gate, token-batched.
        gpu.launch_typed(
            pk.qkv_split,
            [cfg.head_dim.div_ceil(64), cfg.num_heads, n],
            [64, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&cfg.num_heads.to_le_bytes()),
                KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Buffer(pb.q_full),
                KernelArg::Buffer(pb.q),
                KernelArg::Buffer(pb.gate),
            ],
        )?;
        // Per-head q/k norm, in place (each element is read by exactly
        // the thread that rewrites it, so aliasing in/out is safe).
        self.rms_rows(n * cfg.num_heads, cfg.head_dim, 128, pb.q, l.q_norm, pb.q, stream)?;
        self.rms_rows(n * cfg.num_kv_heads, cfg.head_dim, 128, pb.k, l.k_norm, pb.k, stream)?;
        // RoPE (in place), token-batched via grid z.
        let half_dim = cfg.rotary_dim / 2;
        for (heads, buf) in [(cfg.num_heads, pb.q), (cfg.num_kv_heads, pb.k)] {
            self.gpu.launch_typed(
                self.kernels.rope,
                [1, heads, n],
                [half_dim, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&n.to_le_bytes()),
                    KernelArg::Bytes(&heads.to_le_bytes()),
                    KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
                    KernelArg::Bytes(&cfg.rotary_dim.to_le_bytes()),
                    KernelArg::Buffer(pos_ptr),
                    KernelArg::Buffer(bufs.inv_freq),
                    KernelArg::Buffer(buf),
                ],
            )?;
        }
        gpu.launch_typed(
            pk.kvap,
            [cfg.head_dim.div_ceil(64), cfg.num_kv_heads, n],
            [64, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&cfg.num_kv_heads.to_le_bytes()),
                KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
                KernelArg::Bytes(&base_pos.to_le_bytes()),
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Buffer(pb.k),
                KernelArg::Buffer(pb.v),
                KernelArg::Buffer(kv.k),
                KernelArg::Buffer(kv.v),
            ],
        )?;
        let seq_len = base_pos + n;
        let scale: f32 = 1.0 / (cfg.head_dim as f32).sqrt();
        gpu.launch_typed(
            pk.attn,
            [n * cfg.num_heads, 1, 1],
            [128, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Bytes(&seq_len.to_le_bytes()),
                KernelArg::Bytes(&base_pos.to_le_bytes()),
                KernelArg::Bytes(&cfg.num_heads.to_le_bytes()),
                KernelArg::Bytes(&cfg.num_kv_heads.to_le_bytes()),
                KernelArg::Bytes(&cfg.head_dim.to_le_bytes()),
                KernelArg::Bytes(&scale.to_le_bytes()),
                KernelArg::Buffer(pb.q),
                KernelArg::Buffer(kv.k),
                KernelArg::Buffer(kv.v),
                KernelArg::Buffer(pb.attn),
            ],
        )?;
        // Attention output gate (in place on attn).
        let n_el = n * cfg.q_only();
        gpu.launch_typed(
            pk.sigmoid,
            [n_el.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&n_el.to_le_bytes()),
                KernelArg::Buffer(pb.gate),
                KernelArg::Buffer(pb.attn),
                KernelArg::Buffer(pb.attn),
            ],
        )?;
        let xh = self.cast_half(pk, pb, pb.attn, n, cfg.q_only(), stream)?;
        l.o_proj.gemm(gpu, n, pb.attn, xh, pb.o, stream)?;
        self.add_rms_rows(n, x_mat, pb.o, l.post_ln, pb, stream)?;
        self.ffn_batched(pk, pb, &l.gate_proj, &l.up_proj, &l.down_proj, x_mat, n, stream)
    }

    fn lin_layer_batched(
        &self,
        pk: &PrefillKernels,
        pb: &PrefillBufs,
        l: &MetalLinLayer,
        state: &crate::forward::qwen3_5::LinearAttentionState,
        x_mat: DevicePtr,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let gpu = self.gpu.as_ref();
        self.rms_rows(n, cfg.hidden, 512, x_mat, l.input_ln, pb.x_norm, stream)?;
        let xh = self.cast_half(pk, pb, pb.x_norm, n, cfg.hidden, stream)?;
        l.in_proj_a.gemm(gpu, n, pb.x_norm, xh, pb.dt, stream)?;
        l.in_proj_b.gemm(gpu, n, pb.x_norm, xh, pb.b, stream)?;
        l.in_proj_qkv.gemm(gpu, n, pb.x_norm, xh, pb.qkv, stream)?;
        l.in_proj_z.gemm(gpu, n, pb.x_norm, xh, pb.z, stream)?;

        let qkv_total = cfg.qkv_total_lin();
        let qk_channels = 2 * cfg.num_k_heads_lin * cfg.k_head_dim_lin;
        let l2_eps: f32 = 1e-6;
        let block_x = cfg.k_head_dim_lin;
        gpu.launch_typed(
            pk.conv,
            [qkv_total.div_ceil(block_x) * n, 1, 1],
            [block_x, 1, 1],
            0,
            stream,
            &[
                KernelArg::Buffer(state.conv1d_state),
                KernelArg::Buffer(pb.qkv),
                KernelArg::Buffer(l.conv1d),
                KernelArg::Buffer(pb.qkv_smooth),
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Bytes(&qkv_total.to_le_bytes()),
                KernelArg::Bytes(&cfg.conv_kernel_size.to_le_bytes()),
                KernelArg::Bytes(&qk_channels.to_le_bytes()),
                KernelArg::Bytes(&cfg.k_head_dim_lin.to_le_bytes()),
                KernelArg::Bytes(&l2_eps.to_le_bytes()),
            ],
        )?;
        // Advance the conv state past the tile (runs after the sliding
        // reads above — serial encoder order).
        gpu.launch_typed(
            pk.conv_state,
            [qkv_total.div_ceil(128), 1, 1],
            [128, 1, 1],
            0,
            stream,
            &[
                KernelArg::Buffer(state.conv1d_state),
                KernelArg::Buffer(pb.qkv),
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Bytes(&qkv_total.to_le_bytes()),
                KernelArg::Bytes(&cfg.conv_kernel_size.to_le_bytes()),
            ],
        )?;
        let heads = cfg.num_state_heads();
        let n_hb = n * heads;
        gpu.launch_typed(
            pk.gate_beta,
            [n_hb.div_ceil(64), 1, 1],
            [64, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&heads.to_le_bytes()),
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Buffer(pb.dt),
                KernelArg::Buffer(l.dt_bias),
                KernelArg::Buffer(l.a_log),
                KernelArg::Buffer(pb.b),
                KernelArg::Buffer(pb.gatef),
                KernelArg::Buffer(pb.betaf),
            ],
        )?;
        // Chunked GDN: one dispatch per value head, state in registers.
        gpu.launch_typed(
            pk.gdn,
            [cfg.num_v_heads_lin, 1, 1],
            [128, 1, 1],
            0,
            stream,
            &[
                KernelArg::Buffer(state.gdn_state),
                KernelArg::Buffer(pb.qkv_smooth),
                KernelArg::Buffer(pb.gatef),
                KernelArg::Buffer(pb.betaf),
                KernelArg::Buffer(pb.y),
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Bytes(&cfg.num_k_heads_lin.to_le_bytes()),
                KernelArg::Bytes(&cfg.num_v_heads_lin.to_le_bytes()),
                KernelArg::Bytes(&qkv_total.to_le_bytes()),
            ],
        )?;
        self.rms_rows(n * heads, cfg.v_head_dim_lin, 128, pb.y, l.norm_w, pb.y_norm, stream)?;
        // silu(z) ⊙ y_norm, in place on y_norm.
        let n_el = n * cfg.z_dim_lin();
        gpu.launch_typed(
            pk.silu,
            [n_el.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&n_el.to_le_bytes()),
                KernelArg::Buffer(pb.z),
                KernelArg::Buffer(pb.y_norm),
                KernelArg::Buffer(pb.y_norm),
            ],
        )?;
        let xh = self.cast_half(pk, pb, pb.y_norm, n, cfg.z_dim_lin(), stream)?;
        l.out_proj.gemm(gpu, n, pb.y_norm, xh, pb.o, stream)?;
        self.add_rms_rows(n, x_mat, pb.o, l.post_ln, pb, stream)?;
        self.ffn_batched(pk, pb, &l.gate_proj, &l.up_proj, &l.down_proj, x_mat, n, stream)
    }

    /// Fused residual-add + post-norm over `n` rows.
    fn add_rms_rows(
        &self,
        n: u32,
        x_mat: DevicePtr,
        b: DevicePtr,
        post_ln: DevicePtr,
        pb: &PrefillBufs,
        stream: u64,
    ) -> Result<()> {
        self.gpu.launch_typed(
            self.kernels.add_rms,
            [n, 1, 1],
            [512, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&self.cfg.hidden.to_le_bytes()),
                KernelArg::Bytes(&self.cfg.rms_eps.to_le_bytes()),
                KernelArg::Buffer(x_mat),
                KernelArg::Buffer(b),
                KernelArg::Buffer(post_ln),
                KernelArg::Buffer(pb.x_resid),
                KernelArg::Buffer(pb.x_norm2),
            ],
        )
    }

    /// Batched SwiGLU FFN tail; writes the new residual rows into
    /// `x_mat` (the next layer's input).
    #[allow(clippy::too_many_arguments)]
    fn ffn_batched(
        &self,
        pk: &PrefillKernels,
        pb: &PrefillBufs,
        gate_w: &MetalQw,
        up_w: &MetalQw,
        down_w: &MetalQw,
        x_mat: DevicePtr,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        let gpu = self.gpu.as_ref();
        let xh = self.cast_half(pk, pb, pb.x_norm2, n, self.cfg.hidden, stream)?;
        gate_w.gemm(gpu, n, pb.x_norm2, xh, pb.gate_act, stream)?;
        up_w.gemm(gpu, n, pb.x_norm2, xh, pb.up_act, stream)?;
        let n_el = n * self.cfg.intermediate;
        gpu.launch_typed(
            pk.silu,
            [n_el.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&n_el.to_le_bytes()),
                KernelArg::Buffer(pb.gate_act),
                KernelArg::Buffer(pb.up_act),
                KernelArg::Buffer(pb.up_act),
            ],
        )?;
        let xh = self.cast_half(pk, pb, pb.up_act, n, self.cfg.intermediate, stream)?;
        down_w.gemm_resid(gpu, n, xh, pb.x_resid, x_mat, stream)
    }
}
