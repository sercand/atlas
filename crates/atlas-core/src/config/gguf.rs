// SPDX-License-Identifier: AGPL-3.0-only

//! Build a [`ModelConfig`] from GGUF file metadata.
//!
//! GGUF carries its model config inline as metadata key/values
//! (`llama.block_count`, `qwen3.attention.head_count`, …) rather than a
//! sibling `config.json`. This module reads those keys through the
//! [`GgufMeta`] accessor (implemented by the GGUF parser in spark-runtime, so
//! atlas-core keeps no GGUF dependency) and produces a validated
//! [`ModelConfig`] for the llama / qwen2 / qwen3 / gemma decoder families.
//!
//! Strategy: synthesize an HF-config-shaped JSON object from the GGUF keys and
//! deserialize it into `ModelConfig` (serde `#[serde(default)]` fills the many
//! fields GGUF has no analog for), then set the architecture flags
//! (`attn_gated`, `weight_prefix`, gemma `embed_scale` /
//! `final_logit_softcapping`) explicitly, then run the shared
//! [`super::finalize_config`]. No silent production defaults: every value GGUF
//! omits is either derived by an explicit documented rule or is an error.

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};

use super::{ModelConfig, finalize_config};

/// Typed read access to GGUF metadata. Implemented by the spark-runtime GGUF
/// parser over its parsed key/value table. All getters return `None` when the
/// key is absent or holds a different value type — the builder decides whether
/// absence is fatal or has a derivation rule.
pub trait GgufMeta {
    /// Any unsigned/signed integer metadata value, widened to u64.
    fn get_u64(&self, key: &str) -> Option<u64>;
    /// Any float metadata value (f32/f64), widened to f64.
    fn get_f64(&self, key: &str) -> Option<f64>;
    /// A string metadata value.
    fn get_str(&self, key: &str) -> Option<&str>;
    /// Length of an array metadata value (e.g. `tokenizer.ggml.tokens`).
    fn get_arr_len(&self, key: &str) -> Option<usize>;
    /// An integer-array metadata value, widened to u64 (e.g. the MRoPE
    /// `{arch}.rope.dimension_sections`). `None` when the key is absent, is not
    /// an array, or holds an element that is not a non-negative integer.
    fn get_arr_u64(&self, key: &str) -> Option<Vec<u64>>;
}

/// Inputs to [`config_from_gguf`]: the metadata accessor plus two facts the
/// builder needs from the tensor section (not the metadata KV block).
pub struct GgufConfigInputs<'a> {
    pub meta: &'a dyn GgufMeta,
    /// Rows of `token_embd.weight` — the authoritative vocab size when the
    /// `{arch}.vocab_size` key is absent. `None` if the loader could not read
    /// the tensor shape before building the config.
    pub token_embd_vocab: Option<usize>,
    /// Whether the file contains an `output.weight` tensor. Its presence means
    /// an untied LM head; its absence means the LM head ties to the input
    /// embeddings. GGUF has no explicit `tie_word_embeddings` key, so this is
    /// the only reliable signal.
    pub has_output_weight: bool,
}

/// Map a GGUF `general.architecture` string to an Atlas `model_type` (must be
/// a supported loader string) and whether attention Q is gated.
///
/// Plain-decoder GGUFs (llama/qwen2) have no dedicated Atlas arch loader; the
/// closest dense GQA path is the Mistral loader. qwen3 dense maps to `qwen3_5`
/// with `num_experts == 0` (dense qwen3.5 loader). Returns an error for
/// unmapped architectures rather than guessing.
fn arch_to_model_type(arch: &str) -> Result<(&'static str, bool)> {
    // (model_type, attn_gated)
    Ok(match arch {
        "llama" => ("mistral", false),
        // qwen2 ships QKV biases; the Mistral GQA path is the closest dense
        // loader. (Bias handling is a known caveat — see module notes.)
        "qwen2" => ("mistral", false),
        // qwen3 dense: q_norm/k_norm, ungated Q. num_experts==0 → dense loader.
        "qwen3" => ("qwen3_5", false),
        "qwen3moe" => ("qwen3_5_moe", false),
        // Qwen3.5/3.6/3.8 GDN-hybrid (llama.cpp `qwen35` converter). Q IS
        // gated: the full-attention `attn_q` emits 2 · n_head · head_dim rows
        // (q ‖ output gate) — verified on Qwen3.8-27B-Q6_K, whose
        // `blk.3.attn_q.weight` is [5120, 12288] for 24 heads × 256 head_dim.
        // The hybrid / GDN / MTP geometry these arches also need is filled by
        // `apply_qwen35_hybrid` below; `num_experts` picks dense vs MoE.
        "qwen35" | "qwen3_5" => ("qwen3_5", true),
        "qwen35moe" => ("qwen3_5_moe", true),
        // gemma family: GeGLU, ungated Q, embedding scale + logit softcap.
        "gemma" | "gemma2" | "gemma3" | "gemma4" => ("gemma4", false),
        other => bail!(
            "GGUF general.architecture '{other}' has no Atlas model_type mapping. \
             Supported GGUF arches: llama, qwen2, qwen3, qwen3moe, qwen35, qwen35moe, \
             gemma/gemma2/gemma3/gemma4."
        ),
    })
}

/// Fill the Qwen3.5-family hybrid fields (GDN linear-attention geometry, the
/// full/linear layer schedule, the MTP head count and the partial-rotary
/// fraction) into the synthesized config `body`.
///
/// llama.cpp's `qwen35` converter reuses generic Mamba key names for what
/// Qwen3.5 calls Gated DeltaNet, so the mapping is not name-for-name. Verified
/// against `unsloth/Qwen3.8-27B-NVFP4`'s `config.json` (left) and the
/// `orcarouter/Qwen3.8-27B-Uncensored-GGUF` metadata block (right):
///
/// | HF `text_config`         | GGUF `qwen35.*`      | Qwen3.8-27B |
/// |--------------------------|----------------------|-------------|
/// | `linear_num_key_heads`   | `ssm.group_count`    | 16          |
/// | `linear_num_value_heads` | `ssm.time_step_rank` | 48          |
/// | `linear_key_head_dim`    | `ssm.state_size`     | 128         |
/// | `linear_value_head_dim`  | `ssm.inner_size / num_v_heads` | 128 |
/// | `linear_conv_kernel_dim` | `ssm.conv_kernel`    | 4           |
/// | `full_attention_interval`| `full_attention_interval` | 4     |
/// | `mtp_num_hidden_layers`  | `nextn_predict_layers`    | 1     |
/// | `partial_rotary_factor`  | `rope.dimension_count / head_dim` | 0.25 |
///
/// The two head-dim spellings deliberately differ: `state_size` is the shared
/// Q/K head width, while the value width must be divided out of `inner_size`
/// (the fused V span). Both are 128 here, but they are independent fields in HF
/// and conflating them would silently mis-size the V region of `in_proj_qkv`.
///
/// `layer_types` is derived from `full_attention_interval` rather than read:
/// GGUF has no per-layer list. The rule (every `interval`-th layer, 1-based, is
/// full attention) reproduces HF's 64-entry list for Qwen3.8-27B exactly — 16
/// `full_attention` at indices 3, 7, …, 63 and 48 `linear_attention` — which is
/// also what the tensor names say (48 `blk.N.attn_qkv`, 16 `blk.N.attn_q` over
/// the main stack). It is written explicitly instead of left to the runtime
/// fallback so `validate_config`'s length check actually covers it.
fn apply_qwen35_hybrid(
    body: &mut Map<String, Value>,
    meta: &dyn GgufMeta,
    arch: &str,
    num_hidden_layers: usize,
    nextn_layers: usize,
    head_dim: usize,
) -> Result<()> {
    let k = |suffix: &str| format!("{arch}.{suffix}");
    let req = |suffix: &str| -> Result<usize> {
        meta.get_u64(&k(suffix))
            .map(|v| v as usize)
            .with_context(|| {
                format!(
                    "GGUF arch '{arch}' is Qwen3.5-family but required hybrid key \
                     '{arch}.{suffix}' is missing"
                )
            })
    };

    let num_k_heads = req("ssm.group_count")?;
    let num_v_heads = req("ssm.time_step_rank")?;
    let key_head_dim = req("ssm.state_size")?;
    let inner_size = req("ssm.inner_size")?;
    let conv_kernel = req("ssm.conv_kernel")?;
    if num_v_heads == 0 || !inner_size.is_multiple_of(num_v_heads) {
        bail!(
            "GGUF '{arch}.ssm.inner_size' ({inner_size}) is not a multiple of \
             '{arch}.ssm.time_step_rank' ({num_v_heads}); cannot derive linear_value_head_dim"
        );
    }
    body.insert("linear_num_key_heads".into(), json!(num_k_heads));
    body.insert("linear_num_value_heads".into(), json!(num_v_heads));
    body.insert("linear_key_head_dim".into(), json!(key_head_dim));
    body.insert(
        "linear_value_head_dim".into(),
        json!(inner_size / num_v_heads),
    );
    body.insert("linear_conv_kernel_dim".into(), json!(conv_kernel));

    // Layer schedule. `full_attention_interval` is required: without it every
    // layer would default to full attention and the 48 GDN layers would be
    // loaded as attention layers whose tensors do not exist.
    let interval = req("full_attention_interval")?;
    if interval == 0 {
        bail!("GGUF '{arch}.full_attention_interval' is 0; must be >= 1");
    }
    let layer_types: Vec<&str> = (0..num_hidden_layers)
        .map(|i| {
            if (i + 1).is_multiple_of(interval) {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    body.insert("full_attention_interval".into(), json!(interval));
    body.insert("layer_types".into(), json!(layer_types));

    // MTP head. `nextn_predict_layers` was already subtracted out of
    // `num_hidden_layers`; here it becomes the predictor count HF reports.
    body.insert("mtp_num_hidden_layers".into(), json!(nextn_layers));

    // Partial rotary: llama.cpp stores the ROTATED WIDTH in
    // `rope.dimension_count` (64 on Qwen3.8-27B) where HF stores the FRACTION
    // of head_dim (0.25 = 64/256). Absent ⇒ leave serde's 1.0 (full RoPE).
    if let Some(rot) = meta.get_u64(&k("rope.dimension_count")) {
        if head_dim == 0 {
            bail!("GGUF: head_dim is 0; cannot derive partial_rotary_factor");
        }
        body.insert(
            "partial_rotary_factor".into(),
            json!(rot as f64 / head_dim as f64),
        );
    }
    Ok(())
}

/// Build a validated [`ModelConfig`] from GGUF metadata.
pub fn config_from_gguf(inputs: &GgufConfigInputs) -> Result<ModelConfig> {
    let meta = inputs.meta;

    let arch = meta
        .get_str("general.architecture")
        .context("GGUF metadata missing required key 'general.architecture'")?
        .to_string();
    let (model_type, attn_gated) = arch_to_model_type(&arch)?;

    // Namespaced key helper: `{arch}.<suffix>`.
    let k = |suffix: &str| format!("{arch}.{suffix}");
    let req_u64 = |suffix: &str| -> Result<u64> {
        meta.get_u64(&k(suffix))
            .with_context(|| format!("GGUF metadata missing required key '{arch}.{suffix}'"))
    };

    // ── Core dimensions (required) ──
    let hidden_size = req_u64("embedding_length")? as usize;
    let block_count = req_u64("block_count")? as usize;
    // MTP / "next-N" predictor blocks. llama.cpp COUNTS them in `block_count`
    // (Qwen3.8-27B: block_count 65 = 64 main + 1 nextn, whose tensors live at
    // `blk.64.*` including `blk.64.nextn.*`), whereas HF's `num_hidden_layers`
    // counts only the main stack and reports the predictor separately as
    // `mtp_num_hidden_layers`. Subtract so both agree — otherwise `layer_types`
    // is one entry too long, the KV cache is sized for a layer that never runs,
    // and the MTP block is decoded as if it were main-stack layer 64.
    let nextn_layers = meta
        .get_u64(&k("nextn_predict_layers"))
        .map(|v| v as usize)
        .unwrap_or(0);
    let num_hidden_layers = block_count.checked_sub(nextn_layers).with_context(|| {
        format!(
            "GGUF: '{arch}.nextn_predict_layers' ({nextn_layers}) exceeds \
             '{arch}.block_count' ({block_count})"
        )
    })?;
    let intermediate_size = req_u64("feed_forward_length")? as usize;
    let num_attention_heads = req_u64("attention.head_count")? as usize;

    // GQA: kv head count defaults to full MHA (== attention heads) when the key
    // is absent, which is the ggml convention.
    let num_key_value_heads = meta
        .get_u64(&k("attention.head_count_kv"))
        .map(|v| v as usize)
        .unwrap_or(num_attention_heads);

    // head_dim: explicit key_length if present, else hidden_size / head_count.
    // Erroring on a non-divisible fallback avoids a silently-wrong head_dim.
    let head_dim = match meta.get_u64(&k("attention.key_length")) {
        Some(v) => v as usize,
        None => {
            if num_attention_heads == 0 || !hidden_size.is_multiple_of(num_attention_heads) {
                bail!(
                    "GGUF: cannot derive head_dim — '{arch}.attention.key_length' absent and \
                     hidden_size ({hidden_size}) not divisible by head_count ({num_attention_heads})"
                );
            }
            hidden_size / num_attention_heads
        }
    };

    // vocab_size: explicit key → token_embd rows → tokenizer token list length.
    let vocab_size = meta
        .get_u64(&k("vocab_size"))
        .map(|v| v as usize)
        .or(inputs.token_embd_vocab)
        .or_else(|| meta.get_arr_len("tokenizer.ggml.tokens"))
        .context(
            "GGUF: could not determine vocab_size (no '{arch}.vocab_size', no token_embd rows, \
             no 'tokenizer.ggml.tokens')",
        )?;

    // ── Normalization / RoPE / context (documented explicit defaults) ──
    // rms_norm_eps: ggml default is 1e-5 when the key is absent (differs from
    // Atlas's 1e-6 default — we set it explicitly rather than inherit).
    let rms_norm_eps = meta
        .get_f64(&k("attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-5);
    // rope_theta: ggml default 10000.0.
    let rope_theta = meta.get_f64(&k("rope.freq_base")).unwrap_or(10_000.0);
    // context_length is required for a usable KV cache upper bound.
    let max_position_embeddings = req_u64("context_length")? as usize;

    // Tokenizer special tokens (0 when unset is acceptable).
    let bos_token_id = meta.get_u64("tokenizer.ggml.bos_token_id").unwrap_or(0);
    let eos_token_id = meta.get_u64("tokenizer.ggml.eos_token_id").unwrap_or(0);

    // Tied embeddings: no `output.weight` tensor ⇒ tied.
    let tie_word_embeddings = !inputs.has_output_weight;

    // ── MoE (only for MoE arches) ──
    let num_experts = meta
        .get_u64(&k("expert_count"))
        .map(|v| v as usize)
        .unwrap_or(0);

    let mut body: Map<String, Value> = json!({
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "rms_norm_eps": rms_norm_eps,
        "rope_theta": rope_theta,
        "max_position_embeddings": max_position_embeddings,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "tie_word_embeddings": tie_word_embeddings,
        "model_type": model_type,
    })
    .as_object()
    .expect("json! object literal")
    .clone();

    if num_experts > 0 {
        let experts_per_tok = req_u64("expert_used_count").with_context(|| {
            format!("GGUF: MoE arch '{arch}' has expert_count>0 but no '{arch}.expert_used_count'")
        })? as usize;
        let moe_ffn = req_u64("expert_feed_forward_length").with_context(|| {
            format!("GGUF: MoE arch '{arch}' missing '{arch}.expert_feed_forward_length'")
        })? as usize;
        body.insert("num_experts".into(), json!(num_experts));
        body.insert("num_experts_per_tok".into(), json!(experts_per_tok));
        body.insert("moe_intermediate_size".into(), json!(moe_ffn));
    }

    // sliding_window (gemma hybrid attention); 0/absent ⇒ full attention.
    if let Some(sw) = meta.get_u64(&k("attention.sliding_window")) {
        body.insert("sliding_window".into(), json!(sw));
    }

    // Qwen3.5-family hybrid geometry (GDN linear-attention layers, MTP head,
    // partial rotary). Written into `body` so serde fills the typed fields.
    let is_qwen35 = matches!(model_type, "qwen3_5" | "qwen3_5_moe") && arch.starts_with("qwen35");
    if is_qwen35 {
        apply_qwen35_hybrid(
            &mut body,
            meta,
            &arch,
            num_hidden_layers,
            nextn_layers,
            head_dim,
        )?;
    }

    // ── Deserialize numeric fields, then set arch fields explicitly ──
    let raw = Value::Object(body);
    let json_str = serde_json::to_string(&raw).context("serialize synthesized GGUF config")?;
    let mut config: ModelConfig =
        serde_json::from_str(&json_str).context("deserialize synthesized GGUF config")?;

    config.model_type = model_type.to_string();
    config.attn_gated = attn_gated;
    // The GGUF name map emits HF names under the `model.` prefix
    // (`model.embed_tokens.weight`, `model.layers.N.*`, `model.norm.weight`).
    // `layer_prefix()` yields `model.layers.N` for both "" and "model", but the
    // embed/norm/lm_head lookups use the raw prefix — so it must be "model", not
    // "" (else they resolve to `.embed_tokens.weight` and fail).
    config.weight_prefix = "model".to_string();

    // MRoPE is `#[serde(skip)]` (it never comes from JSON), so it must be set
    // here. `{arch}.rope.dimension_sections` is llama.cpp's spelling of HF's
    // `rope_parameters.mrope_section`, emitted 4-wide with a trailing 0
    // ([11, 11, 10, 0] on Qwen3.8-27B) against HF's 3-wide [11, 11, 10].
    // Qwen3.5-family MRoPE is always the interleaved (round-robin) layout;
    // a section list summing to 0 leaves scalar RoPE, matching the default.
    if is_qwen35
        && let Some(sections) = meta.get_arr_u64(&k("rope.dimension_sections"))
        && sections.len() >= 3
    {
        config.mrope_section = [
            sections[0] as usize,
            sections[1] as usize,
            sections[2] as usize,
        ];
        config.mrope_interleaved = config.mrope_section.iter().sum::<usize>() > 0;
    }

    // Gemma-specific post-parse fixups.
    if model_type == "gemma4" {
        config.embed_scale = (hidden_size as f32).sqrt();
        // Logit softcap: honor the GGUF key if present (gemma2), else 0.0
        // (disabled). gemma3+ dropped softcapping.
        config.final_logit_softcapping = meta
            .get_f64(&k("final_logit_softcapping"))
            .map(|v| v as f32)
            .unwrap_or(0.0);
    }

    // Reuse the shared quantization-config + validation pass.
    finalize_config(&mut config, &raw)?;
    Ok(config)
}

// ── Fields GGUF does NOT provide, and how they are set (explicit, no silent
//    prod defaults) ──
//   * partial_rotary_factor / rotary_dim: left at struct default 1.0 (full
//     RoPE) for the plain decoder families. The Qwen3.5 family DOES need it and
//     derives it from `{arch}.rope.dimension_count / head_dim` — see
//     `apply_qwen35_hybrid`.
//   * layer_types / hybrid + SSM fields: left empty for the homogeneous decoder
//     families; populated for `qwen35`/`qwen35moe` by `apply_qwen35_hybrid`.
//   * MLA/DeepSeek/MiniMax/vision fields: 0 / empty — not applicable to the
//     llama/qwen/gemma decoder families this builder targets. A qwen35 GGUF
//     backbone is served TEXT-ONLY: the vision tower is a separate `*mmproj*
//     .gguf`, and `config.vision` is only set by the config.json path, so a
//     bare-GGUF Qwen3.5-VL serves its language model and ignores images.
//   * ep_rank/ep_world_size/tp_*: set at runtime by the caller, not here.

#[cfg(test)]
mod tests;
