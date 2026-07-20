// SPDX-License-Identifier: AGPL-3.0-only

//! Per-token forward pass: CPU embed-row lookup → 64-layer chain through
//! the vendor-agnostic `forward::qwen3_5` orchestration → final norm →
//! packed LM-head GEMV → logits row.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, KernelArg};
use spark_runtime::weights::gguf_q1::{self, Q1_BLOCK_BYTES, Q1_GROUP};

use crate::forward::qwen3_5::{self, FullAttentionLayer, LinearAttentionLayer};

use super::{ForwardBufs, MetalGgufModel, MetalLayer, SlotState};

impl MetalGgufModel {
    /// Embed one token into the residual stream buffer. Text tokens
    /// dequantize their packed Q1 embedding row on the CPU (UMA — the
    /// table never exists in BF16 anywhere); `<|image_pad|>` tokens
    /// splice the next encoded vision row instead.
    fn embed_token(&self, bufs: &mut ForwardBufs, st: &mut SlotState, token: u32) -> Result<()> {
        let hidden = self.cfg.hidden as usize;
        if let Some(v) = &self.vision
            && token == v.pad_token_id
            && let Some(rows) = &mut st.vision
        {
            if rows.cursor >= rows.rows {
                bail!(
                    "more <|image_pad|> tokens than encoded vision rows ({})",
                    rows.rows
                );
            }
            let row_ptr = rows.buf.offset(rows.cursor * v.out_hidden * 2);
            rows.cursor += 1;
            return self
                .gpu
                .copy_d2d(row_ptr, bufs.x_buf, v.out_hidden.min(hidden) * 2);
        }
        if token >= self.cfg.vocab {
            bail!("token id {token} out of vocab range {}", self.cfg.vocab);
        }
        let row_bytes = (hidden / Q1_GROUP) * Q1_BLOCK_BYTES;
        let start = token as usize * row_bytes;
        let row = self
            .embed_host
            .get(start..start + row_bytes)
            .context("embedding row out of bounds")?;
        gguf_q1::dequant_row_f32(row, hidden, &mut bufs.embed_f32)?;
        for (dst, &v) in bufs
            .embed_bf16
            .chunks_exact_mut(2)
            .zip(bufs.embed_f32.iter())
        {
            dst.copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
        }
        self.gpu.copy_h2d(&bufs.embed_bf16, bufs.x_buf)
    }

    /// Stage a prompt sub-chunk's embedding rows into `bufs.x_stage`
    /// and its MRoPE triples into `bufs.pos_stage` — one host upload
    /// for the whole sub-chunk, so the per-token layer loop that
    /// follows issues no host writes. Text rows CPU-dequant into the
    /// host mirror; `<|image_pad|>` rows splice their encoded vision
    /// row with a stream-ordered blit after the upload.
    ///
    /// Caller must drain still-queued GPU readers of the staging
    /// buffers (`synchronize`) before calling — the upload is a plain
    /// UMA memcpy.
    pub(super) fn stage_prefill_inputs(
        &self,
        bufs: &mut ForwardBufs,
        st: &mut SlotState,
        tokens: &[u32],
        pos3: &[[u32; 3]],
        stream: u64,
    ) -> Result<()> {
        let hidden = self.cfg.hidden as usize;
        let row_bytes = hidden * 2;
        let n = tokens.len();
        debug_assert!(n <= bufs.stage_cap && n == pos3.len());

        // (staged row index, encoded vision row ptr) splices, resolved
        // in token order so the cursor walk matches `embed_token`'s.
        let mut splices: Vec<(usize, DevicePtr)> = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            let dst = &mut bufs.stage_host[i * row_bytes..(i + 1) * row_bytes];
            if let Some(v) = &self.vision
                && tok == v.pad_token_id
                && let Some(rows) = &mut st.vision
            {
                if rows.cursor >= rows.rows {
                    bail!(
                        "more <|image_pad|> tokens than encoded vision rows ({})",
                        rows.rows
                    );
                }
                splices.push((i, rows.buf.offset(rows.cursor * v.out_hidden * 2)));
                rows.cursor += 1;
                dst.fill(0);
                continue;
            }
            if tok >= self.cfg.vocab {
                bail!("token id {tok} out of vocab range {}", self.cfg.vocab);
            }
            let packed_row_bytes = (hidden / Q1_GROUP) * Q1_BLOCK_BYTES;
            let start = tok as usize * packed_row_bytes;
            let row = self
                .embed_host
                .get(start..start + packed_row_bytes)
                .context("embedding row out of bounds")?;
            gguf_q1::dequant_row_f32(row, hidden, &mut bufs.embed_f32)?;
            for (d, &v) in dst.chunks_exact_mut(2).zip(bufs.embed_f32.iter()) {
                d.copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        for (i, p3) in pos3.iter().enumerate() {
            for (c, p) in p3.iter().enumerate() {
                bufs.pos_host[i * 12 + c * 4..i * 12 + c * 4 + 4]
                    .copy_from_slice(&p.to_le_bytes());
            }
        }
        self.gpu
            .copy_h2d(&bufs.stage_host[..n * row_bytes], bufs.x_stage)?;
        self.gpu.copy_h2d(&bufs.pos_host[..n * 12], bufs.pos_stage)?;
        let vision_hidden = self.vision.as_ref().map_or(hidden, |v| v.out_hidden);
        for (i, src) in splices {
            self.gpu.copy_d2d_async(
                src,
                bufs.x_stage.offset(i * row_bytes),
                vision_hidden.min(hidden) * 2,
                stream,
            )?;
        }
        Ok(())
    }

    /// Run one token through every decoder layer, leaving the final
    /// residual stream in `bufs.x_buf`. `cache_pos` is the PHYSICAL KV
    /// slot (sequence index); `pos3` is the MRoPE (t, h, w) triple,
    /// which equals `[p, p, p]` for text but diverges from `cache_pos`
    /// after an image run.
    ///
    /// Decode-path entry: the embed/position host uploads at the top
    /// race any still-queued GPU reads of the same buffers, so callers
    /// rely on the sampler's read having drained the previous step
    /// (and `decode_batch` drains between sequences).
    pub(super) fn run_token(
        &self,
        bufs: &mut ForwardBufs,
        st: &mut SlotState,
        token: u32,
        cache_pos: u32,
        pos3: [u32; 3],
        stream: u64,
    ) -> Result<()> {
        self.embed_token(bufs, st, token)?;
        let mut pos_bytes = [0u8; 12];
        for (i, p) in pos3.iter().enumerate() {
            pos_bytes[i * 4..i * 4 + 4].copy_from_slice(&p.to_le_bytes());
        }
        self.gpu.copy_h2d(&pos_bytes, bufs.positions)?;
        self.run_layers(bufs, st, bufs.x_buf, bufs.positions, cache_pos, stream)
    }

    /// The 64-layer chain from an arbitrary layer-0 input row (`x_in`)
    /// and positions pointer, leaving the final residual in
    /// `bufs.x_buf`. Issues no host writes — safe to call in a loop
    /// without synchronizing (the prefill path feeds it staged rows).
    pub(super) fn run_layers(
        &self,
        bufs: &ForwardBufs,
        st: &SlotState,
        x_in: DevicePtr,
        positions: DevicePtr,
        cache_pos: u32,
        stream: u64,
    ) -> Result<()> {
        let pos = cache_pos;
        let mut x = x_in;
        for (idx, layer) in self.layers.iter().enumerate() {
            // Submit the encoded prefix every 16 layers so the GPU
            // works while the host encodes the rest of the token
            // (ordering across command buffers is hazard-tracked).
            if idx > 0 && idx.is_multiple_of(16) {
                self.gpu.flush(stream)?;
            }
            match layer {
                MetalLayer::Full(l) => {
                    let kv = &st.kv[self.kv_ord[idx].expect("kv ordinal for full layer")];
                    let view = FullAttentionLayer {
                        input_ln: l.input_ln,
                        q_norm: l.q_norm,
                        k_norm: l.k_norm,
                        post_ln: l.post_ln,
                        q_proj: &l.q_proj,
                        k_proj: &l.k_proj,
                        v_proj: &l.v_proj,
                        o_proj: &l.o_proj,
                        gate_proj: &l.gate_proj,
                        up_proj: &l.up_proj,
                        down_proj: &l.down_proj,
                    };
                    let out = qwen3_5::forward_full_attention(
                        self.gpu.as_ref(),
                        &self.cfg,
                        &self.kernels,
                        &view,
                        &bufs.full_scratch,
                        kv,
                        bufs.inv_freq,
                        positions,
                        x,
                        pos,
                        pos + 1,
                        stream,
                    )
                    .with_context(|| format!("layer {idx} (full attention)"))?;
                    // full_scratch.x_out aliases x_buf (init), so the
                    // layer already wrote the residual stream in place.
                    if out != bufs.x_buf {
                        self.gpu.copy_d2d_async(
                            out,
                            bufs.x_buf,
                            self.cfg.hidden as usize * 2,
                            stream,
                        )?;
                    }
                    x = bufs.x_buf;
                }
                MetalLayer::Linear(l) => {
                    let state = &st.lin[self.lin_ord[idx].expect("lin ordinal for GDN layer")];
                    let view = LinearAttentionLayer {
                        input_ln: l.input_ln,
                        a_log: l.a_log,
                        dt_bias: l.dt_bias,
                        conv1d_weight: l.conv1d,
                        in_proj_a: &l.in_proj_a,
                        in_proj_b: &l.in_proj_b,
                        in_proj_qkv: &l.in_proj_qkv,
                        in_proj_z: &l.in_proj_z,
                        norm_weight: l.norm_w,
                        out_proj: &l.out_proj,
                        post_ln: l.post_ln,
                        gate_proj: &l.gate_proj,
                        up_proj: &l.up_proj,
                        down_proj: &l.down_proj,
                    };
                    x = qwen3_5::forward_linear_attention(
                        self.gpu.as_ref(),
                        &self.cfg,
                        &self.kernels,
                        &view,
                        state,
                        &bufs.lin_scratch,
                        x,
                        bufs.x_buf,
                        stream,
                        None,
                    )
                    .with_context(|| format!("layer {idx} (GDN)"))?;
                }
            }
        }
        // The next token's embed overwrites x_buf, so make sure the final
        // residual lives there regardless of which scratch buffer the last
        // layer returned.
        if x != bufs.x_buf {
            self.gpu
                .copy_d2d_async(x, bufs.x_buf, self.cfg.hidden as usize * 2, stream)?;
        }
        Ok(())
    }

    /// Final RMSNorm + packed LM-head GEMV from the residual row at
    /// `x` into logits row `row` (`[vocab]` BF16). Does NOT
    /// synchronize — the consumers do (`argmax_of` commits + waits;
    /// `copy_logits_to_host` syncs the default stream before reading).
    pub(super) fn write_logits(
        &self,
        bufs: &ForwardBufs,
        x: DevicePtr,
        row: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        self.gpu.launch_typed(
            self.kernels.rms,
            [1, 1, 1],
            [512, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&self.cfg.hidden.to_le_bytes()),
                KernelArg::Bytes(&self.cfg.rms_eps.to_le_bytes()),
                KernelArg::Buffer(x),
                KernelArg::Buffer(self.final_norm),
                KernelArg::Buffer(bufs.x_final),
            ],
        )?;
        self.lm_head
            .gemv(self.gpu.as_ref(), bufs.x_final, row, stream)
    }

    /// On-device argmax over one `[vocab]` BF16 logits row.
    pub(super) fn argmax_of(&self, logits_row: DevicePtr, stream: u64) -> Result<u32> {
        self.gpu.launch_typed(
            self.argmax,
            [1, 1, 1],
            [512, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&self.cfg.vocab.to_le_bytes()),
                KernelArg::Buffer(logits_row),
                KernelArg::Buffer(self.argmax_out),
            ],
        )?;
        self.gpu.synchronize(stream)?;
        let mut buf = [0u8; 4];
        self.gpu.copy_d2h(self.argmax_out, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }
}
