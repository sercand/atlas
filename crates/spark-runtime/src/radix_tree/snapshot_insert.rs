// SPDX-License-Identifier: AGPL-3.0-only

//! Snapshot-index insert paths (plain, tail, tail-sibling). Split from
//! `snapshot.rs` (file-size cap); same `SsmSnapshotIndex` impl.

use super::snapshot::{SnapshotEntry, SsmSnapshotIndex};

/// The slot an entry being displaced hands back for the caller to free — or
/// `None` if it holds no HBM slot.
///
/// A `tiered` entry's `snapshot_id` is STALE: `evict_to_tier` already returned
/// that slot to the caller (`TierEvict::Spill { slot, .. }`) and the caller
/// already freed it; the entry stayed in the index only as a fault-in record.
/// Handing the same id back a second time is a double-free into
/// `SsmSnapshotPool::free`, whose free list is a plain `Vec` push with no
/// membership check — so the slot is handed to TWO sequences, which then share
/// one GDN/conv state buffer. That is silent cross-stream corruption, not a
/// crash: the same class of fault `ssm_pool::claim_specific` exists to prevent.
///
/// Every other consumer of the index already honours this — `lookup` skips
/// tiered entries ("no HBM slot"), and both victim scans pass
/// `skip_tiered = true`. The insert paths were the one place that did not.
fn freeable_slot(entry: &SnapshotEntry) -> Option<usize> {
    (!entry.tiered).then_some(entry.snapshot_id)
}

impl SsmSnapshotIndex {
    pub(super) fn insert(
        &mut self,
        prefix_hash: u64,
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
    ) -> Option<usize> {
        for entry in &mut self.entries {
            if entry.prefix_hash == prefix_hash {
                let old = freeable_slot(entry);
                entry.snapshot_id = snapshot_id;
                entry.session_hash = session_hash;
                entry.token_count = token_count;
                // A fresh HBM save re-homes the prefix: it is resident again.
                //
                // ⚠ The tier blob under this key is now UNREACHABLE AND NOT
                // RECLAIMED. An earlier version of this comment claimed it was
                // "left to the store's own budget"; that is false on three of
                // the four store arms and the claim is withdrawn:
                //
                //   * the DEFAULT arm is `MemBlobStore::new(0)` — `cap_bytes
                //     == 0` means UNBOUNDED (`selectors.rs`, `store.rs`), so
                //     there is no budget to leave it to;
                //   * a capped `MemBlobStore` evicts FIFO by INSERTION order,
                //     not by coldness, so an orphan is not "the coldest thing
                //     there" under any policy it implements;
                //   * `ArenaSnapshotStore`/`RdmaSnapshotStore` have NO eviction
                //     — slots return only via `remove`, so orphans permanently
                //     consume arena slots and eventually kill the tier.
                //
                // Only `UnifiedSnapshotStore` (default OFF) behaves as claimed.
                // This index layer holds no store handle, so it is structurally
                // incapable of removing the blob; the two `store.remove` call
                // sites are both reap paths in `spark-model`. Tracked as a
                // follow-up — the leak is PRE-EXISTING, but a wrong bound in a
                // comment is worse than none, because it stops the next reader
                // looking.
                entry.tiered = false;
                // A plain save re-homing this prefix is by definition NOT a
                // tail. Without this, an overwrite could re-home another
                // session's is_tail entry (new session_hash, is_tail still
                // set), breaching the <=1-leased-entry-per-session invariant
                // insert_tail's supersede sweep maintains.
                entry.is_tail = false;
                entry.is_tail_sibling = false;
                // A plain save re-homing this prefix is durable — it must not
                // be swept by the decode-checkpoint ring.
                entry.is_decode_ckpt = false;
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                return old;
            }
        }
        self.access_counter += 1;
        self.stats.saves += 1;
        self.entries.push(SnapshotEntry {
            snapshot_id,
            session_hash,
            token_count,
            prefix_hash,
            last_access: self.access_counter,
            tiered: false,
            is_tail: false,
            is_tail_sibling: false,
            is_decode_ckpt: false,
        });
        None
    }

    /// Insert the per-session TAIL snapshot, superseding this session's previous one.
    /// Returns displaced snapshot_ids for the caller to free.
    pub(super) fn insert_tail(
        &mut self,
        prefix_hash: u64,
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
    ) -> Vec<usize> {
        let mut displaced = Vec::new();
        if session_hash != 0 {
            let mut i = 0;
            while i < self.entries.len() {
                if (self.entries[i].is_tail || self.entries[i].is_tail_sibling)
                    && self.entries[i].session_hash == session_hash
                {
                    displaced.extend(freeable_slot(&self.entries.swap_remove(i)));
                } else {
                    i += 1;
                }
            }
        }
        for entry in &mut self.entries {
            if entry.prefix_hash == prefix_hash {
                displaced.extend(freeable_slot(entry));
                entry.snapshot_id = snapshot_id;
                entry.session_hash = session_hash;
                entry.token_count = token_count;
                // Re-homed to HBM, same as the plain `insert` path. This was
                // the ONE overwrite arm that never cleared the flag: an entry
                // left `tiered` while holding a live slot is skipped by
                // `lookup` and by both victim scans, so the slot is reachable
                // by nothing and freeable by nothing — a permanent leak of a
                // scarce snapshot-pool slot, on top of `lookup_tiered` then
                // faulting in bytes that no longer describe this entry.
                entry.tiered = false;
                entry.is_tail = true;
                entry.is_tail_sibling = false;
                entry.is_decode_ckpt = false;
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                return displaced;
            }
        }
        self.access_counter += 1;
        self.entries.push(SnapshotEntry {
            snapshot_id,
            session_hash,
            token_count,
            prefix_hash,
            last_access: self.access_counter,
            tiered: false,
            is_tail: true,
            is_tail_sibling: false,
            is_decode_ckpt: false,
        });
        displaced
    }

    /// Insert the tail's EARLY sibling (`tb - bs`). MUST be called after
    /// [`Self::insert_tail`] within the same finalize — the tail insert's
    /// supersede sweep clears the session's previous tail AND sibling, so
    /// this insert never needs (and must not run) its own sweep.
    pub(super) fn insert_tail_sibling(
        &mut self,
        prefix_hash: u64,
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
    ) -> Option<usize> {
        for entry in &mut self.entries {
            if entry.prefix_hash == prefix_hash {
                let old = freeable_slot(entry);
                entry.snapshot_id = snapshot_id;
                entry.session_hash = session_hash;
                entry.token_count = token_count;
                entry.tiered = false;
                entry.is_tail = false;
                entry.is_tail_sibling = true;
                entry.is_decode_ckpt = false;
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                return old;
            }
        }
        self.access_counter += 1;
        self.stats.saves += 1;
        self.entries.push(SnapshotEntry {
            snapshot_id,
            session_hash,
            token_count,
            prefix_hash,
            last_access: self.access_counter,
            tiered: false,
            is_tail: false,
            is_tail_sibling: true,
            is_decode_ckpt: false,
        });
        None
    }

    /// Insert a DECODE-time Marconi checkpoint, keeping at most `keep`
    /// decode-checkpoint entries for this session (the ring): after the
    /// insert, this session's shallowest `is_decode_ckpt` entries beyond
    /// `keep` are removed. Returns displaced snapshot ids for the caller to
    /// free.
    ///
    /// Why a ring and not plain `insert`: decode checkpoints fire every few
    /// blocks for the whole response, and each older one is dominated for the
    /// SAME growing sequence by the ones after it — but on a small pool the
    /// unswept stream LRU-evicts the prefill prompt-boundary anchors that
    /// serve the next warm turn when the re-rendered prompt diverges at the
    /// generation seam (the common agentic case). The ring bounds a turn's
    /// decode footprint to `keep` slots while still covering warm matches
    /// that land a few blocks below the finish leaf.
    ///
    /// Non-tail semantics: state is captured at exactly `token_count`, so the
    /// entry is content-addressed and safe cross-session (see the `is_tail`
    /// invariant in `snapshot.rs`).
    pub(super) fn insert_decode_ckpt(
        &mut self,
        prefix_hash: u64,
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
        keep: usize,
    ) -> Vec<usize> {
        let mut displaced = Vec::new();
        let mut re_homed = false;
        for entry in &mut self.entries {
            if entry.prefix_hash == prefix_hash {
                displaced.extend(freeable_slot(entry));
                entry.snapshot_id = snapshot_id;
                entry.session_hash = session_hash;
                entry.token_count = token_count;
                entry.tiered = false;
                entry.is_tail = false;
                entry.is_tail_sibling = false;
                entry.is_decode_ckpt = true;
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                re_homed = true;
                break;
            }
        }
        if !re_homed {
            self.access_counter += 1;
            self.stats.saves += 1;
            self.entries.push(SnapshotEntry {
                snapshot_id,
                session_hash,
                token_count,
                prefix_hash,
                last_access: self.access_counter,
                tiered: false,
                is_tail: false,
                is_tail_sibling: false,
                is_decode_ckpt: true,
            });
        }
        // Ring sweep: while this session holds more than `keep` decode
        // checkpoints, drop the SHALLOWEST (smallest token_count) — deeper
        // ones dominate it for every future lookup of this growing sequence.
        let keep = keep.max(1);
        loop {
            let mut count = 0usize;
            let mut shallowest: Option<(usize, usize)> = None; // (idx, token_count)
            for (i, e) in self.entries.iter().enumerate() {
                if e.is_decode_ckpt && e.session_hash == session_hash {
                    count += 1;
                    if shallowest.is_none_or(|(_, tc)| e.token_count < tc) {
                        shallowest = Some((i, e.token_count));
                    }
                }
            }
            if count <= keep {
                break;
            }
            let (idx, _) = shallowest.expect("count > keep implies at least one entry");
            displaced.extend(freeable_slot(&self.entries.swap_remove(idx)));
        }
        displaced
    }
}
