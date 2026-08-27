// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3.8-Flash-Next (`qwen4_exp`) Multi-Token-Prediction draft proposer.
//!
//! Mirrors [`crate::layers::DeepseekV4MtpHead`] (a reused full model layer as
//! the body, shared embed + LM head, own single-layer KV cache, own metadata
//! slab at `MTP_META_OFFSET`), with the qwen4_exp differences:
//!
//! * The combiner consumes the target's PRE-MIXER 4-stream highway (10240
//!   FP32), not the collapsed hidden. HF `transformers` ignores `mtp.*`
//!   entirely; the math below is the vLLM #53896 / SGLang #36497
//!   `residual_linear_shared` form, cross-checked against mlx-serve's
//!   independent implementation (head parity cos 0.99998 vs a torch render):
//!
//!   ```text
//!   ep = fc_embedding · rms1p(embed[token],  pre_fc_norm_embedding)   # [2560]
//!   hn = rms1p(streams_prev, pre_fc_norm_hidden)                      # ONE RMS over all 10240
//!   x[s] = fc_hidden · hn[s*2560..][..2560] + ep      for s in 0..4   # per-stream proj, ep broadcast
//!   ```
//!
//!   `rms1p` is the offset-from-1 norm (`normed * (1 + w)`) — every plain
//!   RMSNorm on this architecture ships Gemma-convention zero-centred
//!   weights.
//!
//! * The body is one full-attention qwen4_exp layer: gated attention + QSA
//!   indexer + low-rank mHC sites + the head's own 512-expert NVFP4 MoE.
//!   Built with a MIDDLE model-layer index so `decode_inner_hc` runs
//!   hc_pre → attn → hc_post → hc_pre → moe → hc_post on `hc_streams`
//!   without expanding or collapsing.
//!
//! * The collapse is the head's own `mtp.hyper_connection_mixer` (low-rank,
//!   no inject — `use_combine=False`), which also carries the final norm;
//!   the LM head is the target's NVFP4 head via `w4a16_gemv`.
//!
//! * Chained drafts (i ≥ 1) feed on the head's own post-block highway, which
//!   the body left in `hc_streams` row 0 — read (and re-written) by the next
//!   combiner pass.
//!
//! Every draft is re-verified by the target under the greedy
//! longest-prefix-match rule, so a defect here costs acceptance, never a
//! wrong emitted token.

use parking_lot::Mutex;
use std::any::Any;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use crate::layer::{AttnMetadataDev, ForwardContext, LayerState, TransformerLayer};
use crate::layers::mtp_meta::{MTP_META_OFFSET, pack_mtp_attn_meta};
use crate::layers::ops;
use crate::layers::qwen3_attention::{HcHeadWeights, HcWeights};
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_map::{DenseWeight, QuantizedWeight};

/// Per-sequence state (mirrors `DeepseekV4MtpProposerState`).
pub struct Qwen4ExpMtpProposerState {
    pub block_table: Vec<u32>,
    pub seq_len: usize,
    pub last_num_drafted: usize,
    pub body_state: Box<dyn LayerState>,
    /// Deferred QSA-key rewind from `after_verify` (no GPU handle there):
    /// applied at the top of the next `propose`, before `begin_verify_aux`.
    pub pending_aux_kept: Option<usize>,
}

impl ProposerState for Qwen4ExpMtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// The loaded MTP components (weight_loader/qwen4_exp/mtp.rs).
pub struct Qwen4ExpMtpModule {
    /// One full-attention qwen4_exp layer with its own MoE, low-rank mHC
    /// sites (middle: no expand, no collapse) and QSA indexer.
    pub body: Box<dyn TransformerLayer>,
    pub fc_embedding: DenseWeight,
    pub fc_hidden: DenseWeight,
    pub pre_fc_norm_embedding: DenseWeight,
    pub pre_fc_norm_hidden: DenseWeight,
    /// `mtp.hyper_connection_mixer` — low-rank collapse carrying the head's
    /// final norm (no inject branch).
    pub mixer: HcHeadWeights,
}

pub struct Qwen4ExpMtpHead {
    module: Qwen4ExpMtpModule,
    /// Shared token embedding table (BF16), from the target model.
    embed_tokens: DenseWeight,
    /// Shared NVFP4 LM head (the target's main head on this model).
    lm_head_nvfp4: QuantizedWeight,
    /// Minimal HcWeights wrapper so `hc_head_site` dispatches the low-rank
    /// variant. `attn`/`ffn` sites are unused by the head call.
    mixer_hc: HcWeights,
    kv_cache: Mutex<PagedKvCache>,

    rms_norm_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    residual_add_k: KernelHandle,
    hc_head_k: KernelHandle,
    argmax_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    bf16_to_f32_k: KernelHandle,
    f32_to_bf16_k: KernelHandle,
}

impl Qwen4ExpMtpHead {
    pub fn new(
        module: Qwen4ExpMtpModule,
        embed_tokens: DenseWeight,
        lm_head_nvfp4: QuantizedWeight,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        max_seq_len: usize,
    ) -> Result<Self> {
        // Single-layer KV cache in the target's attention geometry; the body
        // was built with `attn_idx = 0`, so its cache-pool index is 0.
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let mtp_num_blocks = max_seq_len / kv_config.block_size + 1;
        let kv_cache = PagedKvCache::new(kv_config, mtp_num_blocks, gpu)?;

        let mixer_hc = HcWeights {
            attn: crate::layers::qwen3_attention::HcSiteWeights {
                hc_fn: DevicePtr::NULL,
                hc_base: DevicePtr::NULL,
                hc_scale: DevicePtr::NULL,
                lowrank: module.mixer.lowrank,
            },
            ffn: crate::layers::qwen3_attention::HcSiteWeights {
                hc_fn: DevicePtr::NULL,
                hc_base: DevicePtr::NULL,
                hc_scale: DevicePtr::NULL,
                lowrank: module.mixer.lowrank,
            },
            head: Some(HcHeadWeights {
                hc_fn: DevicePtr::NULL,
                hc_base: DevicePtr::NULL,
                hc_scale: DevicePtr::NULL,
                lowrank: module.mixer.lowrank,
            }),
            hc_mult: config.hc_mult,
            sinkhorn_iters: 0,
            hc_eps: config.rms_norm_eps as f32,
            is_first_model_layer: false,
            is_last_model_layer: true,
        };

        Ok(Self {
            module,
            embed_tokens,
            lm_head_nvfp4,
            mixer_hc,
            kv_cache: Mutex::new(kv_cache),
            // qwen4_exp ships Gemma-convention zero-centred plain norms —
            // the offset-from-1 kernel ("norm"/"rms_norm") is the right one.
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            hc_head_k: gpu.kernel("hyper_connection", "hc_head")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            bf16_to_f32_k: gpu.kernel("residual_add", "bf16_to_f32")?,
            // `relu_squared::convert_f32_to_bf16` is not compiled for this
            // target; the quantizer module's truncating cast is, and the
            // combiner's RMS immediately renormalizes — truncation vs RNE is
            // sub-ULP noise here.
            f32_to_bf16_k: gpu.kernel("quantize_nvfp4", "f32_to_bf16_trunc")?,
        })
    }

    pub fn alloc_state_inner(&self, gpu: &dyn GpuBackend) -> Result<Qwen4ExpMtpProposerState> {
        Ok(Qwen4ExpMtpProposerState {
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
            body_state: self.module.body.alloc_state(gpu)?,
            pending_aux_kept: None,
        })
    }

    /// One draft step. `target_streams` is a [hc_mult * hidden] FP32 highway
    /// row (the target's pre-mixer streams for draft 0; the head's own for
    /// chained drafts). Returns the drafted token id; leaves the head's
    /// post-block highway in `hc_streams` row 0 for chaining.
    #[allow(clippy::too_many_arguments)]
    fn forward_one(
        &self,
        token: u32,
        target_streams: DevicePtr,
        position: usize,
        state: &mut Qwen4ExpMtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let h = ctx.config.hidden_size as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let hc_mult = ctx.config.hc_mult as u32;
        let hw = (hc_mult * h) as usize; // 10240
        let row_bytes = h as usize * 2;

        // Scratch layout inside the SSM buffers (untouched by the
        // attention-type body): streams_bf16 @0, hn @32K, x_bf16 @64K.
        let deint = ctx.buffers.ssm_deinterleaved();
        let streams_bf16 = deint;
        let hn = deint.offset(32768);
        let x_bf16 = deint.offset(65536);

        // ── 1. Embed the input token ──
        let embed_out = ctx.buffers.ssm_qkvz();
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        ctx.gpu.copy_d2d_async(src, embed_out, row_bytes, stream)?;

        // ── 2. Combiner ──
        let normed_embed = ctx.buffers.ssm_gates();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            embed_out,
            &self.module.pre_fc_norm_embedding,
            normed_embed,
            1,
            h,
            eps,
            stream,
        )?;
        let ep = ctx.buffers.ssm_ba();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed_embed,
            &self.module.fc_embedding,
            ep,
            h,
            h,
            stream,
        )?;

        // FP32 highway row → BF16 (read BEFORE hc_streams is overwritten:
        // for chained drafts `target_streams` IS hc_streams row 0).
        KernelLaunch::new(ctx.gpu, self.f32_to_bf16_k)
            .grid([(hw as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(target_streams)
            .arg_ptr(streams_bf16)
            .arg_u32(hw as u32)
            .launch(stream)?;
        // ONE RMS over the whole 10240 vector (weight [10240]).
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            streams_bf16,
            &self.module.pre_fc_norm_hidden,
            hn,
            1,
            hw as u32,
            eps,
            stream,
        )?;
        // Per-stream fc_hidden + broadcast ep.
        for s in 0..hc_mult as usize {
            let x_s = x_bf16.offset(s * row_bytes);
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                hn.offset(s * row_bytes),
                &self.module.fc_hidden,
                x_s,
                h,
                h,
                stream,
            )?;
            ops::residual_add(ctx.gpu, self.residual_add_k, x_s, ep, h, stream)?;
        }
        // BF16 combiner output → the FP32 highway the body reads.
        let hc_streams = ctx.buffers.hc_streams();
        KernelLaunch::new(ctx.gpu, self.bf16_to_f32_k)
            .grid([(hw as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x_bf16)
            .arg_ptr(hc_streams)
            .arg_u32(hw as u32)
            .launch(stream)?;

        // ── 3. Body decode (own KV + QSA state, metadata at MTP_META_OFFSET) ──
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let blocks_needed = (state.seq_len / bs) + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }
        let meta_base = ctx.buffers.scratch().offset(MTP_META_OFFSET);
        let max_blocks = state.block_table.len() as u32;
        let block_idx = state.block_table[state.seq_len / bs];
        let global_slot = (block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64);
        let actual_seq_len = (state.seq_len + 1) as i32;
        let meta_buf = pack_mtp_attn_meta(
            position as u32,
            global_slot,
            actual_seq_len,
            &state.block_table,
            ctx.buffers.scratch_bytes().saturating_sub(MTP_META_OFFSET),
        )?;
        ctx.gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;
        let mtp_meta = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(8),
            seq_len: meta_base.offset(16),
            block_table: meta_base.offset(256),
            max_blocks_per_seq: max_blocks,
            num_seqs: 1,
            seq_slot: DevicePtr(0),
            moe_row_adapter: DevicePtr::NULL,
        };
        let mtp_ctx = ForwardContext {
            buffers: ctx.buffers,
            hc_row_offset: 0,
            gpu: ctx.gpu,
            config: ctx.config,
            dispatch: ctx.dispatch,
            derived: ctx.derived,
            levers: ctx.levers,
            stats: ctx.stats,
            attn_metadata: Some(mtp_meta),
            profile: ctx.profile,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: ctx.token_ids,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        };
        let body_scratch = ctx.buffers.hidden_states();
        let mut disk_block_ids: Vec<u32> = Vec::new();
        let mut disk_last_offloaded: Vec<u32> = vec![0u32; 1];
        let residual = ctx.buffers.residual();
        self.module.body.decode(
            body_scratch,
            residual,
            state.body_state.as_mut(),
            &mut kv_cache,
            state.seq_len,
            &mut state.block_table,
            &mut disk_block_ids,
            &mut disk_last_offloaded,
            &mtp_ctx,
            stream,
        )?;
        drop(kv_cache);

        // ── 4. Mixer collapse (low-rank; carries the head's final norm) ──
        let mixed = ctx.buffers.norm_output();
        ops::hc_head_site(
            ctx.gpu,
            self.hc_head_k,
            hc_streams,
            self.mixer_hc.head.as_ref().expect("mixer set in new()"),
            &self.mixer_hc,
            mixed,
            ctx.buffers.hc_lowrank_scratch(),
            1,
            h,
            eps,
            stream,
        )?;

        // ── 5. Shared NVFP4 LM head → logits → argmax ──
        let v = ctx.config.vocab_size as u32;
        let logits = ctx.buffers.logits();
        ops::w4a16_gemv(
            ctx.gpu,
            self.w4a16_gemv_k,
            mixed,
            &self.lm_head_nvfp4,
            logits,
            v,
            h,
            stream,
        )?;
        let token_id = if let Some(bitmask) = grammar_bitmask {
            argmax_grammar_masked(ctx.gpu, logits, v as usize, bitmask, position)?
        } else {
            let out_ptr = ctx.buffers.scratch();
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out_ptr, v, stream)?;
            let mut buf = [0u8; 4];
            ctx.gpu.copy_d2h(out_ptr, &mut buf)?;
            u32::from_le_bytes(buf)
        };

        state.seq_len += 1;
        Ok(token_id)
    }
}

/// CPU grammar-masked argmax (same contract as `deepseek_v4_mtp`).
fn argmax_grammar_masked(
    gpu: &dyn GpuBackend,
    logits: DevicePtr,
    vocab: usize,
    bitmask: &[i32],
    position: usize,
) -> Result<u32> {
    let mut bf16_buf = vec![0u8; vocab * 2];
    gpu.copy_d2h(logits, &mut bf16_buf)?;
    let mut best_tok = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    let mut any_allowed = false;
    for tok in 0..vocab {
        let word = tok / 32;
        let bit = tok % 32;
        let allowed = word < bitmask.len() && (bitmask[word] & (1i32 << bit)) != 0;
        if !allowed {
            continue;
        }
        any_allowed = true;
        let hi = u16::from_le_bytes([bf16_buf[2 * tok], bf16_buf[2 * tok + 1]]);
        let val = f32::from_bits((hi as u32) << 16);
        if val > best_val {
            best_val = val;
            best_tok = tok as u32;
        }
    }
    if !any_allowed {
        tracing::warn!(
            "qwen4_exp MTP grammar mask allowed zero tokens at pos {position}; \
             returning 0 as pad-draft (rejected at verify)."
        );
        return Ok(0);
    }
    Ok(best_tok)
}

impl DraftProposer for Qwen4ExpMtpHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(self.alloc_state_inner(gpu)?))
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Qwen4ExpMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid qwen4_exp MTP proposer state"))?;
        // Deferred QSA rewind from the previous round's after_verify, then
        // mark this round's base so a partial accept can rewind it too.
        if let Some(kept) = st.pending_aux_kept.take() {
            self.module
                .body
                .rollback_verify_aux(st.body_state.as_mut(), kept, ctx.gpu, stream)?;
        }
        self.module.body.begin_verify_aux(st.body_state.as_mut())?;
        // `target_hidden` is the model's `mtp_hidden_save`, which for this
        // model holds the last token's PRE-MIXER highway row (FP32,
        // hc_mult*hidden — see `save_hidden_for_mtp_dispatch`).
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut current_token = last_token;
        let mut current_streams = target_hidden;
        for i in 0..num_drafts {
            let draft = self.forward_one(
                current_token,
                current_streams,
                position + i,
                st,
                ctx,
                stream,
                grammar_bitmask,
            )?;
            drafts.push(draft);
            current_token = draft;
            // Chained drafts feed on the head's own post-block highway.
            current_streams = ctx.buffers.hc_streams();
        }
        st.last_num_drafted = drafts.len();
        Ok(drafts)
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Qwen4ExpMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid qwen4_exp MTP proposer state"))?;
        let num_drafted = st.last_num_drafted.max(1);
        let num_to_trim = num_drafted.saturating_sub(num_accepted);
        if num_to_trim > 0 {
            st.seq_len = st.seq_len.saturating_sub(num_to_trim);
        }
        // The body's QSA indexer keys are positional and append-only; rewind
        // them alongside the KV rows they mirror. Deferred to the next
        // propose (this hook has no GPU handle; the rewind is host counters).
        st.pending_aux_kept = Some(num_accepted.min(num_drafted));
        Ok(())
    }

    fn free_state(&self, _gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Qwen4ExpMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid qwen4_exp MTP proposer state"))?;
        if !st.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&st.block_table);
            st.block_table.clear();
        }
        st.seq_len = 0;
        Ok(())
    }
}
