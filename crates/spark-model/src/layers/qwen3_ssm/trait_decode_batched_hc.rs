// SPDX-License-Identifier: AGPL-3.0-only

//! K-token single-sequence batched decode (speculative verify) under the mHC
//! highway — the GDN half of what `refuse_batched_under_hc` guarded.
//!
//! Structure mirrors `trait_decode_multi_seq/hc.rs` (N rows, per-seq state):
//! the highway REPLACES the layer's own residual bookkeeping, so the non-hc
//! path's `rms_norm_residual` / `residual_add_rms_norm` / `residual_add`
//! steps must not run. Here the K rows are SEQUENTIAL tokens of ONE
//! sequence, so the recurrence is `batched_gdn_core` (the K-token
//! `decode_batched_conv_gdn` body with per-row intermediates — the same
//! rewind contract `commit_accepted_prefix` consumes), and PLE advances K
//! steps behind a pre-verify snapshot (`verify_snapshot`) so a partial
//! accept can rewind (`rollback_verify`).
//!
//! Per layer:
//!
//!   hc_expand [K]          (first model layer only — the K embedding rows)
//!   PLE snapshot + K-token forward against this sequence's carry
//!   hc_pre  [K] -> mixed rows in norm_output
//!   batched_gdn_core       (per-row h/conv intermediates for the rewind)
//!   hc_post [K] <- out_proj rows (moe_output)
//!   hc_pre  [K] -> mixed rows in norm_output (the MoE input rows)
//!   MoE: forward_k2/k3 at K=2/3 (rows in moe_output), else a per-row loop
//!        staged into `hidden` rows (the embedding rows are dead by then)
//!   hc_post [K]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::Qwen3SsmLayer;
use super::trait_decode_batched::GdnStates;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;

impl Qwen3SsmLayer {
    pub(super) fn decode_batched_inner_hc(
        &self,
        hidden: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
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
        let streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        let normed = ctx.buffers.norm_output();
        let moe_out = ctx.buffers.moe_output();

        // Same hazard as `decode_inner_hc`: this path never runs the fused
        // gate-f32 norm, so FP32 routing would read a stale buffer.
        anyhow::ensure!(
            !self.ffn.fp32_routing_active(),
            "qwen3_ssm mHC batched decode: ATLAS_FP32_ROUTING needs the fused \
             gate-f32 norm, which the highway path replaces. Unset it."
        );

        if hc.is_first_model_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                streams,
                n as u32,
                h as u32,
                hc_mult,
                stream,
            )?;
        }

        // ── PLE: K sequential tokens against this sequence's carry ──
        // Snapshot first so a partial accept can rewind the conv state and
        // the 2-token history (`Model::commit_accepted_prefix`).
        if let Some(ple) = self.ple.as_ref() {
            let host = ctx.host_token_ids.ok_or_else(|| {
                anyhow::anyhow!("hc batched decode: PLE needs host_token_ids threaded")
            })?;
            anyhow::ensure!(
                host.len() >= n,
                "hc batched decode: {} host ids for {n} verify rows",
                host.len()
            );
            let ssm_state = state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
            let st = ssm_state
                .ple
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("PLE verify before prefill: no seq state"))?;
            ple.verify_snapshot(st, &host[..n], ctx.gpu, stream)?;
            ple.forward(st, streams, n, false, ctx, stream)?;
        }

        // ── GDN sublayer ──
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
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
        let out_rows = self.batched_gdn_core(normed, n, GdnStates::Single(state), ctx, stream)?;
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            out_rows,
            streams,
            post,
            comb,
            streams,
            n as u32,
            h as u32,
            stream,
        )?;

        // ── MoE sublayer ──
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
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
                // Per-row loop; stage into `hidden` rows (the embedding rows
                // there are dead once the highway is seeded).
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
            streams,
            post,
            comb,
            streams,
            n as u32,
            h as u32,
            stream,
        )?;
        Ok(())
    }
}
