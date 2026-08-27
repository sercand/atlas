// SPDX-License-Identifier: AGPL-3.0-only

//! PLE Marconi aux-state: serialize / restore the per-sequence lexical
//! carry (token history + conv state) that rides the SSM snapshots.
//! Split from `layer.rs` for the ≤500 LoC cap.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::{PleLayer, PleSeqState};
use crate::layers::ple::ids::ple_ngram_ids;

impl PleLayer {
    /// Marconi aux blob: `[hist_len u32][history u32s][conv f32 bytes]`.
    /// The whole per-sequence carry — a prefix hit restoring KV+SSM without
    /// this would run the n-gram hash on the PREVIOUS request's history.
    pub fn snapshot_aux(
        &self,
        st: &PleSeqState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<u8>> {
        let conv_bytes = self.state_len * self.hc_mult * self.hidden * 4;
        let mut blob = Vec::with_capacity(4 + st.history.len() * 4 + conv_bytes);
        blob.extend_from_slice(&(st.history.len() as u32).to_le_bytes());
        for t in &st.history {
            blob.extend_from_slice(&t.to_le_bytes());
        }
        let off = blob.len();
        blob.resize(off + conv_bytes, 0);
        gpu.copy_d2h_on_stream(st.conv, &mut blob[off..], stream)?;
        Ok(blob)
    }

    /// Restore the blob from [`Self::snapshot_aux`] on a prefix-cache hit.
    pub fn restore_aux(
        &self,
        st: &mut PleSeqState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(blob.len() >= 4, "PLE aux blob truncated");
        let n = u32::from_le_bytes(blob[..4].try_into().unwrap()) as usize;
        let conv_bytes = self.state_len * self.hc_mult * self.hidden * 4;
        anyhow::ensure!(
            blob.len() == 4 + n * 4 + conv_bytes,
            "PLE aux blob size mismatch"
        );
        st.history = blob[4..4 + n * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        st.prestaged_va = None;
        gpu.copy_h2d_async(&blob[4 + n * 4..], st.conv, stream)?;
        Ok(())
    }
}

impl PleLayer {
    /// Snapshot the pre-verify PLE carry (conv state + token history + the K
    /// verify tokens) so a partial accept can rewind. The K-row verify calls
    /// this immediately BEFORE `forward`; `rollback_verify` consumes it.
    pub fn verify_snapshot(
        &self,
        st: &mut PleSeqState,
        tokens: &[u32],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let conv_bytes = self.state_len * self.hc_mult * self.hidden * 4;
        let dst = match st.verify_conv {
            Some(p) => p,
            None => {
                let p = gpu.alloc(conv_bytes)?;
                st.verify_conv = Some(p);
                p
            }
        };
        // A fresh sequence resets `conv`/`history` inside `forward`; mirror
        // that here so the snapshot matches what the forward will consume.
        if st.history.len() != self.dims.context_len() {
            self.reset(st, gpu, stream)?;
        }
        gpu.copy_d2d_async(st.conv, dst, conv_bytes, stream)?;
        st.verify_history = st.history.clone();
        st.verify_tokens = tokens.to_vec();
        Ok(())
    }

    /// Rewind the PLE carry after a partial accept: keep `kept` of the
    /// `verify_tokens.len()` rows the verify forward advanced through.
    ///
    /// * history — last `context_len` of (pre-verify history ++ tokens[..kept])
    /// * conv    — last `state_len` rows of (snapshot ++ gated_normed[..kept]),
    ///   the exact roll `ple_conv` applies, truncated at the accepted row.
    ///   `gated_normed` still holds the verify rows: it is layer-owned scratch
    ///   and the rewind runs before any later forward touches this layer.
    pub fn rollback_verify(
        &self,
        st: &mut PleSeqState,
        kept: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let total = st.verify_tokens.len();
        if total == 0 {
            // No verify forward preceded (this rollback entry point is shared
            // with paths that never ran one), or it was already rolled back.
            return Ok(());
        }
        anyhow::ensure!(
            kept <= total,
            "PLE rollback_verify: kept {kept} > verify rows {total}"
        );
        if kept == total {
            st.verify_tokens.clear();
            return Ok(()); // full accept: the forward's state IS the state
        }
        let snap = st
            .verify_conv
            .ok_or_else(|| anyhow::anyhow!("PLE rollback_verify without a verify_snapshot"))?;
        let c = self.hc_mult * self.hidden;
        let row = c * 4;
        let sl = self.state_len;
        if kept >= sl {
            // New state is entirely verify rows [kept-sl, kept).
            gpu.copy_d2d_async(
                self.gated_normed.offset((kept - sl) * row),
                st.conv,
                sl * row,
                stream,
            )?;
        } else {
            let from_snap = sl - kept;
            gpu.copy_d2d_async(snap.offset(kept * row), st.conv, from_snap * row, stream)?;
            if kept > 0 {
                gpu.copy_d2d_async(
                    self.gated_normed,
                    st.conv.offset(from_snap * row),
                    kept * row,
                    stream,
                )?;
            }
        }
        let mut window = st.verify_history.clone();
        window.extend_from_slice(&st.verify_tokens[..kept]);
        let keep = self.dims.context_len();
        st.history = window[window.len().saturating_sub(keep)..].to_vec();
        st.prestaged_va = None;
        st.verify_tokens.clear();
        Ok(())
    }

    /// Fresh sequence: EOS-filled history and a zeroed conv state.
    pub(super) fn reset(
        &self,
        st: &mut PleSeqState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        st.history = vec![self.dims.eos_token_id; self.dims.context_len()];
        st.prestaged_va = None;
        let zeros = vec![0u8; self.state_len * self.hc_mult * self.hidden * 4];
        gpu.copy_h2d_async(&zeros, st.conv, stream)?;
        Ok(())
    }

    /// Hoisted per-step HOST work for decode under CUDA graphs: the n-gram
    /// hash, the NVMe fault-in and the slot upload into the stable
    /// `slots_dev` buffer. All three are capture-illegal (the upload reads
    /// pageable memory, which invalidates a recording graph with status
    /// 901), so the scheduler calls this BEFORE graph replay/capture — the
    /// same phasing decode_a already gives the `token_ids` upload. `forward`
    /// then consumes `prestaged_va` and enqueues only stable-buffer kernels.
    ///
    /// History advances HERE; the prestaged `forward` must not advance it
    /// again.
    pub fn prestage(
        &self,
        st: &mut PleSeqState,
        tokens: &[u32],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if st.history.len() != self.dims.context_len() {
            self.reset(st, gpu, stream)?;
        }
        let mut window = st.history.clone();
        window.extend_from_slice(tokens);
        let all = ple_ngram_ids(&self.dims, &window);
        let rows = &all[all.len() - tokens.len()..];
        let flat: Vec<u64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let va = self.gather_host(&flat, gpu, stream)?;
        let keep = self.dims.context_len();
        st.history = window[window.len() - keep..].to_vec();
        st.prestaged_va = Some(va);
        st.last_staged_va = va;
        Ok(())
    }
}
