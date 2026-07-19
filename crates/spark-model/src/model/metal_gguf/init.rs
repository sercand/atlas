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

fn load_full_layer(store: &WeightStore, i: usize) -> Result<MetalFullLayer> {
    let p = |s: &str| format!("model.layers.{i}.{s}");
    Ok(MetalFullLayer {
        input_ln: bf16_ptr(store, &p("input_layernorm.weight"))?,
        q_norm: bf16_ptr(store, &p("self_attn.q_norm.weight"))?,
        k_norm: bf16_ptr(store, &p("self_attn.k_norm.weight"))?,
        post_ln: bf16_ptr(store, &p("post_attention_layernorm.weight"))?,
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
        input_ln: bf16_ptr(store, &p("input_layernorm.weight"))?,
        a_log: widen_bf16_to_f32(gpu, a_log_bf16, cfg.num_state_heads() as usize)?,
        dt_bias: bf16_ptr(store, &p("linear_attn.dt_bias"))?,
        conv1d: bf16_ptr(store, &p("linear_attn.conv1d.weight"))?,
        norm_w: bf16_ptr(store, &p("linear_attn.norm.weight"))?,
        post_ln: bf16_ptr(store, &p("post_attention_layernorm.weight"))?,
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

fn alloc_forward_bufs(gpu: &dyn GpuBackend, cfg: &Qwen35ForwardConfig) -> Result<ForwardBufs> {
    let bf16 = |n: u32| -> Result<DevicePtr> { gpu.alloc(n as usize * 2) };
    let f32b = |n: u32| -> Result<DevicePtr> { gpu.alloc(n as usize * 4) };

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
        x_out: bf16(cfg.hidden)?,
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
        x_final: bf16(cfg.hidden)?,
    };

    // Partial-RoPE inv_freq table: rotary_dim/2 entries.
    let half_dim = cfg.rotary_dim / 2;
    let inv_freq_bytes: Vec<u8> = (0..half_dim)
        .map(|i| 1.0f32 / cfg.rope_theta.powf(2.0 * i as f32 / cfg.rotary_dim as f32))
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let inv_freq = gpu.alloc(inv_freq_bytes.len())?;
    gpu.copy_h2d(&inv_freq_bytes, inv_freq)?;

    Ok(ForwardBufs {
        x_buf: bf16(cfg.hidden)?,
        x_final: bf16(cfg.hidden)?,
        positions: gpu.alloc(4)?,
        inv_freq,
        full_scratch,
        lin_scratch,
        embed_f32: vec![0f32; cfg.hidden as usize],
        embed_bf16: vec![0u8; cfg.hidden as usize * 2],
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
    let kernels = Qwen35Kernels::resolve(gpu.as_ref())
        .context("resolving metal kernels (was the binary built with ATLAS_TARGET_HW=metal?)")?;
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
                    load_full_layer(store, i).with_context(|| format!("layer {i} (full)"))?,
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

    let final_norm = bf16_ptr(store, "model.norm.weight")?;
    let fwd = alloc_forward_bufs(gpu.as_ref(), &cfg)?;

    let max_batch = max_batch_size.max(1);
    let logits = gpu.alloc(2 * max_batch * cfg.vocab as usize * 2)?;
    let argmax_out = gpu.alloc(4 * max_batch)?;

    let kv_dtype = metal_kv_dtype(kv_dtype);
    tracing::info!(
        "MetalGgufModel: kv dtype {:?}, max_seq_len {max_seq_len}, max concurrent seqs {max_batch}",
        kv_dtype
    );

    Ok(MetalGgufModel {
        gpu,
        cfg,
        kernels,
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
