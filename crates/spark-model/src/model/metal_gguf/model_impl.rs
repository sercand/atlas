// SPDX-License-Identifier: AGPL-3.0-only

//! `impl Model for MetalGgufModel`. Prefill walks the prompt through the
//! single-token forward; decode is one step per call; `decode_batch`
//! runs sequences serially (the forward is weight-bandwidth-bound, so
//! serial decode with per-slot state is the honest v1 shape).
//! Speculative / MTP / graph paths are not applicable and stub exactly
//! like `NllbGpuModel`'s.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::DevicePtr;

use crate::forward::qwen3_5::{LayerKvCache, LinearAttentionState};
use crate::traits::{Model, SequenceState};

use super::{MetalGgufModel, SlotState};

impl MetalGgufModel {
    /// Run prompt positions `[chunk_start, chunk_start + chunk_len)` and
    /// emit logits into the slot's prefill row when `emit_logits`.
    fn prefill_range(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        emit_logits: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        let end = chunk_start
            .checked_add(chunk_len)
            .context("prefill range overflow")?;
        if end > tokens.len() {
            bail!(
                "prefill chunk [{chunk_start}, {end}) exceeds prompt length {}",
                tokens.len()
            );
        }
        if end as u32 > self.max_seq_len {
            bail!(
                "prompt length {end} exceeds --max-seq-len {}",
                self.max_seq_len
            );
        }

        let mut bufs = self.fwd.lock().expect("forward lock");
        let states = self.states.lock().expect("states lock");
        let st = states
            .get(&seq.slot_idx)
            .context("prefill: no device state for slot (alloc_sequence not called?)")?;
        for (i, &tok) in tokens[chunk_start..end].iter().enumerate() {
            self.run_token(&mut bufs, st, tok, (chunk_start + i) as u32, stream)?;
        }
        let row = self.prefill_logits_row(seq.slot_idx);
        if emit_logits {
            self.write_logits(&bufs, row, stream)?;
        }
        drop(states);
        drop(bufs);

        // Scheduler bookkeeping (mirrors NllbGpuModel: the model owns these).
        if chunk_start == 0 {
            seq.tokens = tokens.to_vec();
            seq.prompt_len = tokens.len();
        }
        seq.seq_len = end;
        seq.kv_valid_tokens = end;
        Ok(if emit_logits { row } else { DevicePtr::NULL })
    }

    fn decode_one(
        &self,
        token: u32,
        seq: &mut SequenceState,
        row: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let pos = seq.seq_len as u32;
        if pos >= self.max_seq_len {
            bail!(
                "sequence length {pos} reached --max-seq-len {}",
                self.max_seq_len
            );
        }
        let mut bufs = self.fwd.lock().expect("forward lock");
        let states = self.states.lock().expect("states lock");
        let st = states
            .get(&seq.slot_idx)
            .context("decode: no device state for slot")?;
        self.run_token(&mut bufs, st, token, pos, stream)?;
        self.write_logits(&bufs, row, stream)?;
        drop(states);
        drop(bufs);
        seq.tokens.push(token);
        seq.seq_len += 1;
        Ok(())
    }
}

impl Model for MetalGgufModel {
    fn prefill(&self, tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        self.prefill_range(tokens, seq, 0, tokens.len(), true, stream)
    }

    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_range(tokens, seq, chunk_start, chunk_len, is_last_chunk, stream)
    }

    fn decode(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        let row = self.decode_logits_row(0);
        self.decode_one(token, seq, row, stream)?;
        Ok(row)
    }

    fn decode_batch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        let n = seqs.len();
        if n == 0 || tokens.len() != n {
            bail!(
                "decode_batch: tokens/seqs length mismatch ({}, {n})",
                tokens.len()
            );
        }
        for (i, seq) in seqs.iter_mut().enumerate() {
            self.decode_one(tokens[i], seq, self.decode_logits_row(i), stream)?;
        }
        Ok(self.decode_logits_row(0))
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab as usize
    }

    fn bind_gpu_to_thread(&self) -> Result<()> {
        self.gpu.bind_to_thread()
    }

    fn alloc_sequence(&self) -> Result<SequenceState> {
        let slot = self
            .free_slots
            .lock()
            .expect("slot lock")
            .pop()
            .context("no free metal sequence slots (raise --max-num-seqs)")?;
        let n_kv = self.kv_ord.iter().flatten().count();
        let n_lin = self.lin_ord.iter().flatten().count();
        let build = || -> Result<SlotState> {
            let kv = (0..n_kv)
                .map(|_| {
                    LayerKvCache::alloc(
                        self.gpu.as_ref(),
                        self.kv_dtype,
                        self.max_seq_len,
                        self.cfg.kv_dim(),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let conv_bytes = (self.cfg.qkv_total_lin() * self.cfg.conv_kernel_size) as usize * 4;
            let gdn_floats = (self.cfg.num_v_heads_lin
                * self.cfg.k_head_dim_lin
                * self.cfg.v_head_dim_lin) as usize;
            let lin = (0..n_lin)
                .map(|_| -> Result<LinearAttentionState> {
                    let conv1d_state = self.gpu.alloc(conv_bytes)?;
                    let gdn_state = self.gpu.alloc(gdn_floats * 4)?;
                    self.gpu.memset(conv1d_state, 0, conv_bytes)?;
                    self.gpu.memset(gdn_state, 0, gdn_floats * 4)?;
                    Ok(LinearAttentionState {
                        conv1d_state,
                        gdn_state,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(SlotState { kv, lin })
        };
        let st = match build() {
            Ok(st) => st,
            Err(e) => {
                self.free_slots.lock().expect("slot lock").push(slot);
                return Err(e).context("allocating per-sequence KV/GDN state");
            }
        };
        self.states.lock().expect("states lock").insert(slot, st);

        Ok(SequenceState {
            adapter_id: 0,
            adapter_slot: -1,
            acquired_adapter_slot: -1,
            src_lang_id: 0,
            tgt_lang_id: 0,
            num_beams: 1,
            length_penalty: 1.0,
            early_stopping: false,
            tokens: Vec::new(),
            block_table: Vec::new(),
            seq_len: 0,
            layer_states: Vec::new(),
            proposer_state: None,
            slot_idx: slot,
            ssm_slot: None,
            marconi_skip_to: 0,
            marconi_exact_snap: None,
            session_hash: 0,
            chunked_prefill_meta: None,
            cached_prefix_tokens: 0,
            kv_valid_tokens: 0,
            last_decode_ckpt_block: 0,
            prompt_len: 0,
            collect_prompt_logprobs: None,
            prompt_logprobs: Vec::new(),
            disk_block_ids: Vec::new(),
            disk_last_offloaded_per_layer: Vec::new(),
        })
    }

    fn free_sequence(&self, seq: &mut SequenceState) -> Result<()> {
        let slot = seq.slot_idx;
        if slot == usize::MAX {
            return Ok(()); // migrated to a survivor by compact_sequence
        }
        if let Some(st) = self.states.lock().expect("states lock").remove(&slot) {
            for kv in st.kv {
                self.gpu.free(kv.k)?;
                self.gpu.free(kv.v)?;
                if let Some(p) = kv.k_scales {
                    self.gpu.free(p)?;
                }
                if let Some(p) = kv.v_scales {
                    self.gpu.free(p)?;
                }
            }
            for lin in st.lin {
                self.gpu.free(lin.conv1d_state)?;
                self.gpu.free(lin.gdn_state)?;
            }
        }
        self.free_slots.lock().expect("slot lock").push(slot);
        Ok(())
    }

    fn compact_sequence(&self, seq: &mut SequenceState, new_slot: usize) -> Result<()> {
        let old = seq.slot_idx;
        let mut map = self.states.lock().expect("states lock");
        if let Some(st) = map.remove(&old) {
            map.insert(new_slot, st);
        }
        seq.slot_idx = new_slot;
        Ok(())
    }

    fn detach_slot_for_reuse(&self, seq: &mut SequenceState) {
        seq.slot_idx = usize::MAX;
    }

    fn cache_sequence(&self, _seq: &SequenceState) {
        // No prefix cache on the metal path yet.
    }

    fn num_free_blocks(&self) -> usize {
        // KV lives in per-slot contiguous caches outside the paged block
        // pool; report ample headroom so admission math never rejects.
        1 << 20
    }

    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.gpu.copy_d2h(logits_ptr, dst)
    }

    fn logits_buffer_ptr(&self) -> DevicePtr {
        self.decode_logits_row(0)
    }

    fn argmax_on_device(&self, logits_ptr: DevicePtr, stream: u64) -> Result<u32> {
        self.argmax_of(logits_ptr, stream)
    }

    fn argmax_batch(&self, logits_ptr: DevicePtr, n: usize, stream: u64) -> Result<Vec<u32>> {
        (0..n)
            .map(|i| self.argmax_of(logits_ptr.offset(i * self.cfg.vocab as usize * 2), stream))
            .collect()
    }

    // ── inapplicable paths: no speculative / SSM-snapshot / MTP here ──

    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }

    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        bail!("metal: speculative verify not supported")
    }

    fn checkpoint_ssm_states(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }

    fn rollback_ssm_states(&self, _seq: &mut SequenceState, _n: usize) -> Result<()> {
        Ok(())
    }

    fn generate_speculative(
        &self,
        _tokens: &[u32],
        _params: &spark_runtime::sampler::SamplingParams,
        _num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult> {
        bail!("metal: speculative decoding not supported")
    }

    fn has_proposer(&self) -> bool {
        false
    }

    fn has_self_speculative(&self) -> bool {
        false
    }

    fn decode_draft(
        &self,
        _token: u32,
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<DevicePtr> {
        bail!("metal: self-speculative draft not supported")
    }

    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        bail!("metal: graphed verify not supported")
    }

    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        bail!("metal: graphed verify not supported")
    }

    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        bail!("metal: graphed verify not supported")
    }

    fn save_hidden_for_mtp(&self, _token_idx: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn run_mtp_propose(
        &self,
        _token: u32,
        _position: usize,
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Option<u32>> {
        Ok(None)
    }

    fn run_mtp_propose_multi(
        &self,
        _token: u32,
        _position: usize,
        _num_drafts: usize,
        _seq: &mut SequenceState,
        _stream: u64,
        _grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        Ok(Vec::new())
    }

    fn trim_proposer_state(
        &self,
        _seq: &mut SequenceState,
        _num_accepted: usize,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }
}
