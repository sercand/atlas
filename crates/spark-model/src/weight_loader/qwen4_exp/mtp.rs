// SPDX-License-Identifier: AGPL-3.0-only

//! Loader for the qwen4_exp MTP draft module (#753 item I, revived).
//!
//! The checkpoint's `mtp.*` namespace is one full-attention decoder layer
//! (its own 512-expert MoE, low-rank mHC sites, QSA indexer) plus a
//! two-branch combiner (`fc_embedding`/`fc_hidden` with their pre-norms) and
//! the head's own `hyper_connection_mixer`. The fused BF16 expert tensors
//! the checkpoint ships are pre-sliced and pre-quantized offline into
//! `extra_weights.safetensors` (per-expert ModelOpt NVFP4, the exact layout
//! `load_moe_qwen35` reads for the 48 main layers), so this loader is the
//! main-layer arms verbatim under the `mtp.layers.0` prefix.
//!
//! The body is attached with a MIDDLE model-layer index: no `hc_expand`, no
//! `hc_head` — the proposer (`layers::qwen4_exp_mtp`) supplies both ends.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layers::qwen4_exp_mtp::Qwen4ExpMtpModule;
use crate::weight_map::dense;

const LP: &str = "mtp.layers.0";

pub fn load_mtp_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<Qwen4ExpMtpModule>> {
    if !store.contains("mtp.fc_embedding.weight") {
        tracing::info!(
            "qwen4_exp: no mtp.* tensors in the store — MTP off. The main \
             shards' fused MTP block is skipped by design; drafting needs the \
             pre-sliced `extra_weights.safetensors` (bench/qwen4_exp docs)."
        );
        return Ok(None);
    }
    let h = config.hidden_size;
    let variant = crate::weight_map::detect_nvfp4_variant(store, config);
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();

    let ffn = super::ffn::build_moe(store, LP, config, gpu, variant)
        .context("qwen4_exp MTP: MoE block")?;
    let input_norm = super::ones_norm(h, gpu)?;
    let post_attn_norm = super::ones_norm(h, gpu)?;
    let mut body =
        crate::weight_loader::qwen35::load_layers::attention_arms::build_full_attention_nvfp4(
            1, // display index only — MIDDLE, so hc neither expands nor collapses
            store,
            LP,
            gpu,
            variant,
            config,
            h,
            absmax_k,
            quantize_k,
            stream,
            KvCacheDtype::Bf16,
            0, // attn_idx: the head's own single-layer KV cache indexes pool 0
            input_norm,
            post_attn_norm,
            ffn,
        )
        .context("qwen4_exp MTP: attention body")?;
    let (attn_site, ffn_site) = super::hc::load_layer_sites(store, LP, config)?;
    super::aux::attach_hc(&mut body, 1, attn_site, ffn_site, None, config)?;
    super::aux::attach_qsa(&mut body, 1, LP, store, config, gpu)?;

    let mixer = super::hc::load_head_at(store, "mtp.hyper_connection_mixer", config)?;
    let g = |name: &str| dense(store, name).with_context(|| format!("qwen4_exp MTP: {name}"));
    tracing::info!(
        "qwen4_exp MTP draft module loaded: 1 full-attention layer + \
         {}-expert MoE + low-rank mHC + QSA indexer, combiner + mixer",
        config.num_experts,
    );
    Ok(Some(Qwen4ExpMtpModule {
        body,
        fc_embedding: g("mtp.fc_embedding.weight")?,
        fc_hidden: g("mtp.fc_hidden.weight")?,
        pre_fc_norm_embedding: g("mtp.pre_fc_norm_embedding.weight")?,
        pre_fc_norm_hidden: g("mtp.pre_fc_norm_hidden.weight")?,
        mixer,
    }))
}
