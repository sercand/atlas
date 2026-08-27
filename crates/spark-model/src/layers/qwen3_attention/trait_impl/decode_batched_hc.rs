// SPDX-License-Identifier: AGPL-3.0-only

//! K-token single-sequence batched decode (speculative verify) for the
//! attention layer under the mHC highway.
//!
//! The verify drivers treat K rows as K virtual sequences and call
//! `decode_multi_seq` with DUMMY states — correct for stateless attention,
//! wrong under QSA: the indexer's per-sequence key history must ingest the K
//! rows IN ORDER into the REAL state (and rewind the rejected tail —
//! `QsaIndexer::rewind_verify_tail`). This path brackets the highway ops over
//! all K rows at once and runs the attention core per row through
//! `attention_forward` — the same single-token path serial decode takes, so
//! QSA ingest/selection, KV write and rope all behave exactly as K sequential
//! decodes — at the cost of re-reading the (NVFP4, 12-layer) attention
//! weights per row. The bandwidth-heavy GDN/MoE layers batch; this is the
//! cheap 25% of the model.
//!
//! Per-row metadata is the caller's K-row upload viewed one row at a time:
//! positions/seq_len at `+t*4`, slot at `+t*8` (i64), block_table at
//! `+t*max_blocks*4`.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{AttnMetadataDev, ForwardContext, LayerState};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_batched_inner_hc(
        &self,
        hidden: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_tokens;
        let bf16 = 2usize;
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decode_batched_inner_hc without mHC weights"))?;
        let hc_mult = hc.hc_mult as u32;
        let hc_streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        let normed = ctx.buffers.norm_output();
        let moe_out = ctx.buffers.moe_output();
        let meta = ctx
            .attn_metadata
            .ok_or_else(|| anyhow::anyhow!("attention verify requires K-row metadata"))?;

        if hc.is_first_model_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                hc_streams,
                n as u32,
                h as u32,
                hc_mult,
                stream,
            )?;
        }

        // ── Attention sublayer ──
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            hc_streams,
            &hc.attn,
            hc,
            normed,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        if ops::HcVariant::of(hc).applies_block_input_norm() {
            // Sinkhorn variant: the block input norm runs after hc_pre.
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                normed,
                &self.input_norm,
                normed,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
        }

        // Mark the verify base so a partial accept can rewind the indexer's
        // key-history counters (`rollback_verify_aux`).
        if let Some(qsa) = self.qsa.as_ref() {
            let st = crate::layers::qwen3_attention::helpers::qsa_seq_state(qsa, state, ctx.gpu)?;
            qsa.begin_verify(st);
        }

        // Per-row attention through the single-token path: row t writes KV at
        // position seq_len + t and (under QSA) ingests its indexer key into
        // the REAL per-sequence state, selection included — sequential rows
        // on one stream see each other's cache writes, exactly like K serial
        // decode steps. Outputs staged into `hidden` rows (the embedding rows
        // are dead once the highway is seeded).
        for t in 0..n {
            let row_meta = AttnMetadataDev {
                positions: meta.positions.offset(t * 4),
                positions_h: meta.positions_h.offset(t * 4),
                positions_w: meta.positions_w.offset(t * 4),
                slot: meta.slot.offset(t * 8),
                seq_len: meta.seq_len.offset(t * 4),
                block_table: meta
                    .block_table
                    .offset(t * meta.max_blocks_per_seq as usize * 4),
                max_blocks_per_seq: meta.max_blocks_per_seq,
                num_seqs: 1,
                seq_slot: meta.seq_slot.offset(t * 4),
                moe_row_adapter: meta.moe_row_adapter,
            };
            let row_ctx = ForwardContext {
                buffers: ctx.buffers,
                hc_row_offset: ctx.hc_row_offset,
                gpu: ctx.gpu,
                config: ctx.config,
                dispatch: ctx.dispatch,
                derived: ctx.derived,
                levers: ctx.levers,
                stats: ctx.stats,
                attn_metadata: Some(row_meta),
                profile: ctx.profile,
                comm: ctx.comm,
                graph_capture: ctx.graph_capture,
                gdn_exact_replay: ctx.gdn_exact_replay,
                token_ids: ctx.token_ids,
                host_token_ids: ctx.host_token_ids,
                routed_lora_layers: None,
                midchunk_capture: None,
                moe_lora_route: ctx.moe_lora_route,
            };
            let attn_out = self.attention_forward(
                state,
                normed.offset(t * h * bf16),
                seq_len + t,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                kv_cache,
                &row_ctx,
                stream,
            )?;
            if ctx.config.tp_world_size > 1
                && let Some(comm) = ctx.comm
            {
                comm.all_reduce_async(attn_out.0, h * bf16, stream)?;
            }
            if let Some(ref post_norm) = self.post_attn_out_norm {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    attn_out,
                    post_norm,
                    attn_out,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
            }
            ctx.gpu
                .copy_d2d_async(attn_out, hidden.offset(t * h * bf16), h * bf16, stream)?;
        }

        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            hidden,
            hc_streams,
            post,
            comb,
            hc_streams,
            n as u32,
            h as u32,
            stream,
        )?;

        // Standalone attention (no FFN)
        if self.ffn.is_none() {
            if hc.is_last_model_layer
                && let Some(ref head) = hc.head
            {
                ops::hc_head_site(
                    ctx.gpu,
                    self.hc_head_k,
                    hc_streams,
                    head,
                    hc,
                    hidden,
                    ctx.buffers.hc_lowrank_scratch(),
                    n as u32,
                    h as u32,
                    eps,
                    stream,
                )?;
            }
            return Ok(());
        }

        // ── FFN sublayer ──
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            hc_streams,
            &hc.ffn,
            hc,
            normed,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        if ops::HcVariant::of(hc).applies_block_input_norm() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                normed,
                &self.post_attn_norm,
                normed,
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
        }
        let moe_rows = match n {
            2 => {
                self.ffn.forward_k2(normed, ctx, stream)?;
                moe_out
            }
            3 => {
                self.ffn.forward_k3(normed, ctx, stream)?;
                moe_out
            }
            _ => {
                for i in 0..n {
                    let out = self.ffn.forward(normed.offset(i * h * bf16), ctx, stream)?;
                    ctx.gpu
                        .copy_d2d_async(out, hidden.offset(i * h * bf16), h * bf16, stream)?;
                }
                hidden
            }
        };
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            moe_rows,
            hc_streams,
            post,
            comb,
            hc_streams,
            n as u32,
            h as u32,
            stream,
        )?;

        if hc.is_last_model_layer
            && let Some(ref head) = hc.head
        {
            ops::hc_head_site(
                ctx.gpu,
                self.hc_head_k,
                hc_streams,
                head,
                hc,
                hidden,
                ctx.buffers.hc_lowrank_scratch(),
                n as u32,
                h as u32,
                eps,
                stream,
            )?;
        }
        Ok(())
    }
}
