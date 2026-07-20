// SPDX-License-Identifier: AGPL-3.0-only

//! `MetalGgufModel` construction: forward-config derivation, per-layer
//! weight wiring from the [`WeightStore`], scratch/logits allocation,
//! and the host-side packed-embedding copy.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;
use spark_runtime::weights::gguf_q1::GgufQ1Weight;

use crate::forward::qwen3_5::{
    FullAttentionScratch, LinearAttentionScratch, Qwen35ForwardConfig, Qwen35Kernels,
};

use super::{
    ForwardBufs, MetalFullLayer, MetalGgufModel, MetalLayer, MetalLinLayer, MetalQw, metal_kv_dtype,
};

/// Derive the vendor-agnostic forward config from the parsed HF config.
fn forward_config(config: &ModelConfig) -> Result<Qwen35ForwardConfig> {
    let rotary_dim = if config.rotary_dim > 0 {
        config.rotary_dim
    } else {
        (config.head_dim as f64 * config.partial_rotary_factor) as usize
    };
    if config.linear_num_key_heads == 0 || config.linear_num_value_heads == 0 {
        bail!(
            "metal GGUF model needs GDN dims in config.json (linear_num_key_heads / \
             linear_num_value_heads)"
        );
    }
    Ok(Qwen35ForwardConfig {
        hidden: config.hidden_size as u32,
        intermediate: config.intermediate_size as u32,
        num_layers: config.num_hidden_layers as u32,
        vocab: config.vocab_size as u32,
        group_size: 128,
        rms_eps: config.rms_norm_eps as f32,
        num_heads: config.num_attention_heads as u32,
        num_kv_heads: config.num_key_value_heads as u32,
        head_dim: config.head_dim as u32,
        rope_theta: config.rope_theta as f32,
        rotary_dim: rotary_dim as u32,
        num_k_heads_lin: config.linear_num_key_heads as u32,
        num_v_heads_lin: config.linear_num_value_heads as u32,
        k_head_dim_lin: config.linear_key_head_dim as u32,
        v_head_dim_lin: config.linear_value_head_dim as u32,
        conv_kernel_size: config.linear_conv_kernel_dim as u32,
    })
}

/// BF16 store tensor → raw device pointer (norms, dt_bias, conv1d, …).
fn bf16_ptr(store: &WeightStore, name: &str) -> Result<DevicePtr> {
    let t = store.get(name)?;
    if t.dtype != spark_runtime::weights::WeightDtype::BF16 {
        bail!("{name}: expected BF16, got {:?}", t.dtype);
    }
    Ok(t.ptr)
}

/// `A_log` arrives BF16 from the GGUF loader but the GDN gate kernel
/// reads FP32 — widen host-side into a fresh F32 device buffer (a few
/// dozen floats per layer).
fn widen_bf16_to_f32(gpu: &dyn GpuBackend, src: DevicePtr, n: usize) -> Result<DevicePtr> {
    let mut raw = vec![0u8; n * 2];
    gpu.copy_d2h(src, &mut raw)?;
    let mut out = Vec::with_capacity(n * 4);
    for chunk in raw.chunks_exact(2) {
        let v = half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
        out.extend_from_slice(&v.to_le_bytes());
    }
    let dst = gpu.alloc(out.len())?;
    gpu.copy_h2d(&out, dst)?;
    Ok(dst)
}

/// RMSNorm weights that carry the Qwen3-Next `+1` offset. The GGUF loader
/// normalizes them to the zero-centered HF form (`w_hf = w_gguf − 1`) for
/// the CUDA kernels, which compute `x·(1+w)/rms`. The Metal `rms_norm`
/// kernel is VANILLA (`x·w/rms` — it was built for MLX checkpoints, which
/// ship vanilla weights), so re-add the 1 here into a fresh BF16 buffer.
/// The GDN `linear_attn.norm` is untouched by the loader and stays raw.
fn norm_plus_one(gpu: &dyn GpuBackend, store: &WeightStore, name: &str) -> Result<DevicePtr> {
    let t = store.get(name)?;
    if t.dtype != spark_runtime::weights::WeightDtype::BF16 {
        bail!("{name}: expected BF16 norm weight, got {:?}", t.dtype);
    }
    let n = t.num_elements();
    let mut raw = vec![0u8; n * 2];
    gpu.copy_d2h(t.ptr, &mut raw)?;
    for chunk in raw.chunks_exact_mut(2) {
        let v = half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32() + 1.0;
        chunk.copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }
    let dst = gpu.alloc(raw.len())?;
    gpu.copy_h2d(&raw, dst)?;
    Ok(dst)
}

fn load_full_layer(store: &WeightStore, gpu: &dyn GpuBackend, i: usize) -> Result<MetalFullLayer> {
    let p = |s: &str| format!("model.layers.{i}.{s}");
    Ok(MetalFullLayer {
        input_ln: norm_plus_one(gpu, store, &p("input_layernorm.weight"))?,
        q_norm: norm_plus_one(gpu, store, &p("self_attn.q_norm.weight"))?,
        k_norm: norm_plus_one(gpu, store, &p("self_attn.k_norm.weight"))?,
        post_ln: norm_plus_one(gpu, store, &p("post_attention_layernorm.weight"))?,
        q_proj: MetalQw::from_store(store, &p("self_attn.q_proj.weight"))?,
        k_proj: MetalQw::from_store(store, &p("self_attn.k_proj.weight"))?,
        v_proj: MetalQw::from_store(store, &p("self_attn.v_proj.weight"))?,
        o_proj: MetalQw::from_store(store, &p("self_attn.o_proj.weight"))?,
        gate_proj: MetalQw::from_store(store, &p("mlp.gate_proj.weight"))?,
        up_proj: MetalQw::from_store(store, &p("mlp.up_proj.weight"))?,
        down_proj: MetalQw::from_store(store, &p("mlp.down_proj.weight"))?,
    })
}

fn load_lin_layer(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    cfg: &Qwen35ForwardConfig,
    i: usize,
) -> Result<MetalLinLayer> {
    let p = |s: &str| format!("model.layers.{i}.{s}");
    let a_log_bf16 = bf16_ptr(store, &p("linear_attn.A_log"))?;
    Ok(MetalLinLayer {
        input_ln: norm_plus_one(gpu, store, &p("input_layernorm.weight"))?,
        a_log: widen_bf16_to_f32(gpu, a_log_bf16, cfg.num_state_heads() as usize)?,
        dt_bias: bf16_ptr(store, &p("linear_attn.dt_bias"))?,
        conv1d: bf16_ptr(store, &p("linear_attn.conv1d.weight"))?,
        norm_w: bf16_ptr(store, &p("linear_attn.norm.weight"))?,
        post_ln: norm_plus_one(gpu, store, &p("post_attention_layernorm.weight"))?,
        in_proj_a: MetalQw::from_store(store, &p("linear_attn.in_proj_a.weight"))?,
        in_proj_b: MetalQw::from_store(store, &p("linear_attn.in_proj_b.weight"))?,
        in_proj_qkv: MetalQw::from_store(store, &p("linear_attn.in_proj_qkv.weight"))?,
        in_proj_z: MetalQw::from_store(store, &p("linear_attn.in_proj_z.weight"))?,
        out_proj: MetalQw::from_store(store, &p("linear_attn.out_proj.weight"))?,
        gate_proj: MetalQw::from_store(store, &p("mlp.gate_proj.weight"))?,
        up_proj: MetalQw::from_store(store, &p("mlp.up_proj.weight"))?,
        down_proj: MetalQw::from_store(store, &p("mlp.down_proj.weight"))?,
    })
}

/// Any packed-Q1 projection in the loader's row-planar byte order?
/// (Planar has no GEMM kernel variant — it gates batched prefill off.)
fn layer_has_planar(l: &MetalLayer) -> bool {
    let p = |q: &MetalQw| matches!(q, MetalQw::Q1(w) if w.planar);
    match l {
        MetalLayer::Full(f) => {
            p(&f.q_proj)
                || p(&f.k_proj)
                || p(&f.v_proj)
                || p(&f.o_proj)
                || p(&f.gate_proj)
                || p(&f.up_proj)
                || p(&f.down_proj)
        }
        MetalLayer::Linear(g) => {
            p(&g.in_proj_a)
                || p(&g.in_proj_b)
                || p(&g.in_proj_qkv)
                || p(&g.in_proj_z)
                || p(&g.out_proj)
                || p(&g.gate_proj)
                || p(&g.up_proj)
                || p(&g.down_proj)
        }
    }
}

fn alloc_forward_bufs(
    gpu: &dyn GpuBackend,
    cfg: &Qwen35ForwardConfig,
    max_seq_len: usize,
    batched_prefill: bool,
) -> Result<ForwardBufs> {
    let bf16 = |n: u32| -> Result<DevicePtr> { gpu.alloc(n as usize * 2) };
    let f32b = |n: u32| -> Result<DevicePtr> { gpu.alloc(n as usize * 4) };

    // The residual stream. Both layer kinds write their final fused
    // GEMV directly into this buffer (x_out / x_final alias it), which
    // saves a per-layer blit — and the encoder round-trip a blit forces.
    let x_buf = bf16(cfg.hidden)?;

    let full_scratch = FullAttentionScratch {
        x_norm: bf16(cfg.hidden)?,
        q_full: bf16(cfg.q_total())?,
        q_split: bf16(cfg.q_only())?,
        gate_split: bf16(cfg.q_only())?,
        k: bf16(cfg.kv_dim())?,
        v: bf16(cfg.kv_dim())?,
        q_norm_out: bf16(cfg.q_only())?,
        k_norm_out: bf16(cfg.kv_dim())?,
        attn_out: bf16(cfg.q_only())?,
        gated_attn: bf16(cfg.q_only())?,
        o: bf16(cfg.hidden)?,
        x_resid: bf16(cfg.hidden)?,
        x_norm2: bf16(cfg.hidden)?,
        gate_act: bf16(cfg.intermediate)?,
        up_act: bf16(cfg.intermediate)?,
        x_out: x_buf,
    };
    let lin_scratch = LinearAttentionScratch {
        x_norm: bf16(cfg.hidden)?,
        dt_raw: bf16(cfg.num_state_heads())?,
        b_raw: bf16(cfg.num_state_heads())?,
        qkv: bf16(cfg.qkv_total_lin())?,
        qkv_smooth: bf16(cfg.qkv_total_lin())?,
        z: bf16(cfg.z_dim_lin())?,
        gate: f32b(cfg.num_state_heads())?,
        beta: f32b(cfg.num_state_heads())?,
        y: bf16(cfg.z_dim_lin())?,
        y_norm: bf16(cfg.z_dim_lin())?,
        out: bf16(cfg.hidden)?,
        x_resid: bf16(cfg.hidden)?,
        x_norm2: bf16(cfg.hidden)?,
        gate_act: bf16(cfg.intermediate)?,
        up_act: bf16(cfg.intermediate)?,
        x_final: x_buf,
    };

    // Partial-RoPE inv_freq table: rotary_dim/2 entries.
    let half_dim = cfg.rotary_dim / 2;
    let inv_freq_bytes: Vec<u8> = (0..half_dim)
        .map(|i| 1.0f32 / cfg.rope_theta.powf(2.0 * i as f32 / cfg.rotary_dim as f32))
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let inv_freq = gpu.alloc(inv_freq_bytes.len())?;
    gpu.copy_h2d(&inv_freq_bytes, inv_freq)?;

    // Prefill staging: up to `stage_cap` tokens of embeddings +
    // positions per upload (10 KB/token at hidden = 5120 — capping at
    // 1024 keeps it ~10 MB while amortizing the per-sub-chunk sync).
    let stage_cap = max_seq_len.clamp(1, 1024);

    Ok(ForwardBufs {
        x_buf,
        x_final: bf16(cfg.hidden)?,
        // One token × 3 u32 MRoPE components (t, h, w).
        positions: gpu.alloc(12)?,
        inv_freq,
        full_scratch,
        lin_scratch,
        embed_f32: vec![0f32; cfg.hidden as usize],
        embed_bf16: vec![0u8; cfg.hidden as usize * 2],
        x_stage: gpu.alloc(stage_cap * cfg.hidden as usize * 2)?,
        pos_stage: gpu.alloc(stage_cap * 12)?,
        stage_cap,
        stage_host: vec![0u8; stage_cap * cfg.hidden as usize * 2],
        pos_host: vec![0u8; stage_cap * 12],
        prefill: if batched_prefill {
            Some(super::prefill::PrefillBufs::alloc(
                gpu,
                cfg,
                super::prefill::PREFILL_TILE.min(stage_cap),
            )?)
        } else {
            None
        },
    })
}

pub(super) fn build(
    config: &ModelConfig,
    store: &WeightStore,
    gpu: Box<dyn GpuBackend>,
    max_seq_len: usize,
    max_batch_size: usize,
    kv_dtype: KvCacheDtype,
) -> Result<MetalGgufModel> {
    let cfg = forward_config(config)?;
    let mut kernels = Qwen35Kernels::resolve(gpu.as_ref())
        .context("resolving metal kernels (was the binary built with ATLAS_TARGET_HW=metal?)")?;
    // The LLM trunk always rotates through the 3-stream interleaved MRoPE
    // kernel; for text tokens all three streams carry the same position,
    // which reproduces scalar `rope_apply` bit-for-bit. The positions
    // buffer is [1, 3] u32 (see `ForwardBufs`).
    kernels.rope = gpu.kernel("rope_mrope_interleaved", "rope_mrope_interleaved")?;
    let argmax = gpu.kernel("argmax_bf16", "argmax_bf16")?;

    // Layer roles + weights.
    let n_layers = config.num_hidden_layers;
    let mut layers = Vec::with_capacity(n_layers);
    let mut kv_ord = vec![None; n_layers];
    let mut lin_ord = vec![None; n_layers];
    let (mut n_kv, mut n_lin) = (0usize, 0usize);
    for i in 0..n_layers {
        match config.layer_type(i) {
            LayerType::FullAttention => {
                kv_ord[i] = Some(n_kv);
                n_kv += 1;
                layers.push(MetalLayer::Full(
                    load_full_layer(store, gpu.as_ref(), i)
                        .with_context(|| format!("layer {i} (full)"))?,
                ));
            }
            LayerType::LinearAttention => {
                lin_ord[i] = Some(n_lin);
                n_lin += 1;
                layers.push(MetalLayer::Linear(
                    load_lin_layer(store, gpu.as_ref(), &cfg, i)
                        .with_context(|| format!("layer {i} (linear)"))?,
                ));
            }
            other => bail!("layer {i}: unsupported layer type {other:?} on metal"),
        }
    }
    tracing::info!(
        "MetalGgufModel: {n_layers} layers ({n_kv} full-attention + {n_lin} GDN), \
         hidden {}, vocab {}",
        cfg.hidden,
        cfg.vocab
    );

    // Embedding: host copy of the packed table for CPU row lookups.
    let embed_t = store.get("model.embed_tokens.weight")?;
    if !embed_t.is_packed_q1() {
        bail!(
            "model.embed_tokens.weight must be keep-packed Q1_0 on metal \
             (native-Q1 load path); got {:?}",
            embed_t.dtype
        );
    }
    let mut embed_host = vec![0u8; embed_t.byte_size()];
    gpu.copy_d2h(embed_t.ptr, &mut embed_host)?;

    // Untied LM head (falls back to the embedding for tied checkpoints).
    let lm_head = if store.contains("lm_head.weight") {
        GgufQ1Weight::from_store(store, "lm_head.weight")?
    } else {
        GgufQ1Weight::from_tensor(embed_t, "model.embed_tokens.weight (tied lm_head)")?
    };

    let final_norm = norm_plus_one(gpu.as_ref(), store, "model.norm.weight")?;

    let kv_dtype = metal_kv_dtype(kv_dtype);
    // Batched prefill: needs the GEMM kernels (blocked byte order), a
    // BF16 KV cache, and register-resident GDN state (128×128 heads).
    let batch_env = std::env::var("ATLAS_METAL_PREFILL_BATCH")
        .map(|v| v != "0")
        .unwrap_or(true);
    let batched_ok = batch_env
        && kv_dtype == crate::forward::qwen3_5::MetalKvDtype::Bf16
        && cfg.k_head_dim_lin == 128
        && cfg.v_head_dim_lin == 128
        && !layers.iter().any(layer_has_planar);
    let prefill_kernels = if batched_ok {
        match super::prefill::PrefillKernels::resolve(gpu.as_ref()) {
            Ok(k) => Some(k),
            Err(e) => {
                tracing::warn!("batched prefill kernels unavailable — per-token prefill: {e:#}");
                None
            }
        }
    } else {
        tracing::info!(
            "batched prefill disabled (env={batch_env}, kv={kv_dtype:?}, planar or non-128 GDN dims)"
        );
        None
    };

    let fwd = alloc_forward_bufs(gpu.as_ref(), &cfg, max_seq_len, prefill_kernels.is_some())?;

    let max_batch = max_batch_size.max(1);
    let logits = gpu.alloc(2 * max_batch * cfg.vocab as usize * 2)?;
    let argmax_out = gpu.alloc(4 * max_batch)?;

    tracing::info!(
        "MetalGgufModel: kv dtype {:?}, max_seq_len {max_seq_len}, max concurrent seqs {max_batch}, \
         batched prefill {}",
        kv_dtype,
        if prefill_kernels.is_some() { "on" } else { "off" }
    );

    // Vision tower: present when the config declares one AND the mmproj
    // sidecar tensors landed in the store.
    let vision = match &config.vision {
        Some(vc) if store.contains("model.visual.patch_embed.proj.weight") => {
            let v = super::vision::MetalVision::from_store(store, gpu.as_ref(), vc)
                .context("building metal vision tower from mmproj tensors")?;
            tracing::info!(
                "MetalGgufModel: vision tower ready ({} blocks, pad token {})",
                vc.depth,
                v.pad_token_id
            );
            Some(v)
        }
        Some(_) => {
            tracing::info!(
                "MetalGgufModel: config declares vision but no mmproj tensors loaded — text-only"
            );
            None
        }
        None => None,
    };

    Ok(MetalGgufModel {
        vision,
        pending_vision: Mutex::new(None),
        gpu,
        cfg,
        kernels,
        prefill_kernels,
        argmax,
        layers,
        kv_ord,
        lin_ord,
        final_norm,
        lm_head,
        embed_host,
        max_seq_len: max_seq_len as u32,
        max_batch,
        kv_dtype,
        fwd: Mutex::new(fwd),
        free_slots: Mutex::new((0..max_batch).rev().collect()),
        states: Mutex::new(HashMap::new()),
        logits,
        argmax_out,
    })
}
