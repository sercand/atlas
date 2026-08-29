// SPDX-License-Identifier: AGPL-3.0-only

//! NVMe-backed row cache for the n-gram embedding tables.
//!
//! The n-gram tables of the LongCat / Qwen3.8-Flash-Next family are the
//! model's largest tensors by far (31.4 B params on LongCat-Flash-Lite,
//! ~51 B announced for Flash-Next) and simultaneously its *least*
//! bandwidth-hungry: a token touches exactly one row per table — 12 rows,
//! ~3 KB — regardless of sequence length. Pure capacity, near-zero
//! bandwidth, which makes them the best demotion candidate in the model.
//!
//! Design, and why it needs no CUDA kernel change:
//!
//! * The cache is a flat PINNED arena of `slots × row_stride` bytes. On
//!   GB10 pinned host memory is GPU-addressable at the SAME virtual address
//!   ([`ExpertArena`] asserts this), so the arena *is* a
//!   `[slots, dim]` device-side table.
//! * The n-gram row ids are computed HOST-side (they are a pure function of
//!   token ids), so a lookup resolves `row_id -> slot` on the host and hands
//!   the gather kernel the SLOT INDEX in place of the row id. `batched_embed`
//!   / `batched_embed_fp8` then run verbatim against the arena base.
//! * A miss reads the row straight off NVMe into its pinned slot — no
//!   `cuMemcpyHtoD` anywhere on the path.
//!
//! Eviction is CLOCK (second-chance): O(1), no per-hit bookkeeping, and it
//! approximates LRU well for the power-law access pattern these tables have.
//! Rows touched by the CURRENT batch are pinned so a large prefill can never
//! evict a row it is still about to read.
//!
//! O_DIRECT requires 4 KiB-aligned reads, while a row is typically 256 B
//! (FP8, dim 256). Reads are therefore issued as the containing 4 KiB block
//! into a bounce buffer and the row copied out — the block is the disk's
//! minimum transfer anyway, so this costs no extra I/O, only a 256 B host
//! memcpy. Cache capacity stays row-granular, which matters because the
//! hash scatters ids: neighbouring rows in a table are unrelated.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::expert_arena::ExpertArena;

/// O_DIRECT transfer granularity (also `ExpertArena`'s stride requirement).
pub(crate) const BLOCK: usize = 4096;

/// One table's on-NVMe backing file plus its resident row cache.
pub struct NgramRowCache {
    /// Flat pinned, GPU-addressable `[slots, row_stride]` region.
    arena: ExpertArena,
    /// Backing files: row `i` at byte offset `base_offset + i * row_stride`
    /// inside `files[0]`, or — for a SEGMENTED table — inside the file its
    /// shard names (see [`Segments::file_of`]).
    ///
    /// `base_offset` lets the cache read STRAIGHT OUT OF A SAFETENSORS SHARD
    /// — a table is already a contiguous row-major blob there, so no repack
    /// or re-save is needed. Because that offset is only 8-byte aligned, a
    /// row may straddle a 4 KiB O_DIRECT block; `fetch_into` handles the seam.
    files: Vec<File>,
    base_offset: u64,
    /// SEGMENTED tables: one base offset per equal-sized shard.
    ///
    /// LongCat ships each n-gram table as ONE contiguous safetensors tensor,
    /// so `base_offset` alone locates every row. Qwen3.8-Flash-Next splits its
    /// single 320M-row table across 128 shard tensors which are NOT laid out
    /// consecutively in the file — the shards interleave with other weights,
    /// so a global row id needs its shard's own base. `None` keeps the
    /// original single-offset behaviour byte for byte.
    segments: Option<Segments>,
    /// Per-row scale file mirror (FP8 tables), `None` for BF16 tables.
    scales: Option<ScaleCache>,
    row_stride: usize,
    slots: usize,
    rows_total: u64,
    /// row_id -> slot.
    map: HashMap<u64, u32>,
    /// slot -> resident row id (`u64::MAX` = empty).
    slot_row: Vec<u64>,
    /// CLOCK reference bits.
    refbit: Vec<bool>,
    /// Slots pinned for the batch in flight (never evicted).
    pinned: Vec<bool>,
    hand: usize,
    bounce: AlignedBlock,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

/// A table split across equal-sized shards at scattered file offsets.
struct Segments {
    /// Byte offset of each shard's first row, indexed by shard.
    bases: Vec<u64>,
    /// Which backing file each shard lives in, as an index into
    /// [`NgramRowCache::files`]. All-zero for the single-file case.
    ///
    /// RadixArk's release is preprocessed into ONE 102.4 GB BF16 file, so the
    /// cache was written to open exactly one. primitive-ai's mixed build
    /// ships the same 128 shards already BF16 but spread over 43
    /// `ple-bf16-*.safetensors` — and it is 102 GB of disk and a repack pass
    /// to flatten that, for a table the cache only ever reads 320 bytes of at
    /// a time. One fd per shard file costs nothing and removes the repack.
    file_of: Vec<u16>,
    /// Rows per shard. Every shard but conceivably the last holds exactly
    /// this many; `open_segmented` requires them all equal so the mapping is
    /// a divide rather than a search.
    rows_per: u64,
}

/// Per-row f32 scales for an FP8 table, mirrored into a device-visible
/// `[slots]` array indexed by SLOT (parallel to the arena).
struct ScaleCache {
    arena: ExpertArena,
    file: File,
}

/// A 4 KiB-aligned host buffer for O_DIRECT reads.
pub(crate) struct AlignedBlock {
    buf: Vec<u8>,
    off: usize,
}

impl AlignedBlock {
    /// Two blocks: a row whose base offset is not 4 KiB-aligned (every row of
    /// a table read in place from a safetensors shard) can straddle one
    /// boundary, and two blocks always cover it since `row_stride <= BLOCK`.
    pub(crate) fn new() -> Self {
        // Over-allocate and take an aligned window (portable, no libc::memalign).
        let buf = vec![0u8; BLOCK * 3];
        let addr = buf.as_ptr() as usize;
        let off = (BLOCK - (addr % BLOCK)) % BLOCK;
        Self { buf, off }
    }
    /// `n` whole blocks of aligned scratch (`n <= 2`).
    pub(crate) fn blocks(&mut self, n: usize) -> &mut [u8] {
        &mut self.buf[self.off..self.off + n * BLOCK]
    }
}

impl NgramRowCache {
    /// Open `path` as the backing store for a table of `rows_total` rows of
    /// `row_stride` bytes, caching `slots` of them in pinned GPU-addressable
    /// memory. `scale_path` supplies the per-row f32 scales of an FP8 table.
    pub fn open(
        path: &Path,
        scale_path: Option<&Path>,
        rows_total: u64,
        row_stride: usize,
        slots: usize,
    ) -> Result<Self> {
        Self::open_at(path, 0, scale_path, rows_total, row_stride, slots)
    }

    /// As [`Self::open`], but the table starts at `base_offset` inside the
    /// file — the safetensors-shard case (`data_offsets[0]` + the header
    /// length), which needs no re-save of the checkpoint.
    #[allow(clippy::too_many_arguments)]
    pub fn open_at(
        path: &Path,
        base_offset: u64,
        scale_path: Option<&Path>,
        rows_total: u64,
        row_stride: usize,
        slots: usize,
    ) -> Result<Self> {
        if row_stride == 0 || slots == 0 {
            bail!("NgramRowCache: zero geometry (row_stride={row_stride}, slots={slots})");
        }
        if row_stride > BLOCK {
            bail!(
                "NgramRowCache: row_stride {row_stride} exceeds the {BLOCK}-byte \
                 O_DIRECT block; a row would span more than the two blocks the \
                 seam-handling fetch reads"
            );
        }
        // One flat pinned region: `slots * row_stride` bytes, rounded up to the
        // arena's 4 KiB stride requirement.
        let bytes = slots * row_stride;
        let blocks = bytes.div_ceil(BLOCK);
        let arena =
            ExpertArena::new(1, blocks as u32, BLOCK).context("NgramRowCache: pinned arena")?;
        let file = open_direct(path)?;
        let scales = match scale_path {
            Some(sp) => {
                let sbytes = slots * 4;
                let sblocks = sbytes.div_ceil(BLOCK);
                Some(ScaleCache {
                    arena: ExpertArena::new(1, sblocks as u32, BLOCK)
                        .context("NgramRowCache: scale arena")?,
                    file: open_direct(sp)?,
                })
            }
            None => None,
        };
        Ok(Self {
            arena,
            files: vec![file],
            base_offset,
            segments: None,
            scales,
            row_stride,
            slots,
            rows_total,
            map: HashMap::with_capacity(slots * 2),
            slot_row: vec![u64::MAX; slots],
            refbit: vec![false; slots],
            pinned: vec![false; slots],
            hand: 0,
            bounce: AlignedBlock::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        })
    }

    /// As [`Self::open_at`], but for a table split across equal-sized shards
    /// at SCATTERED file offsets — Qwen3.8-Flash-Next's PLE table, whose 128
    /// shard tensors are not laid out consecutively inside the safetensors
    /// file. `bases[i]` is shard `i`'s first row; every shard holds
    /// `rows_per_shard` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn open_segmented(
        path: &Path,
        bases: Vec<u64>,
        rows_per_shard: u64,
        scale_path: Option<&Path>,
        row_stride: usize,
        slots: usize,
    ) -> Result<Self> {
        let file_of = vec![0u16; bases.len()];
        Self::open_segmented_multi(
            std::slice::from_ref(&path),
            file_of,
            bases,
            rows_per_shard,
            scale_path,
            row_stride,
            slots,
        )
    }

    /// As [`Self::open_segmented`], but the shards may live in DIFFERENT
    /// files: `file_of[shard]` indexes `paths`.
    ///
    /// A checkpoint that ships its n-gram table as N safetensors shards
    /// spread over M files (primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8:
    /// 128 shards over 43 `ple-bf16-*.safetensors`) is served in place, with
    /// no repack of a 102 GB table into one file.
    #[allow(clippy::too_many_arguments)]
    pub fn open_segmented_multi(
        paths: &[&Path],
        file_of: Vec<u16>,
        bases: Vec<u64>,
        rows_per_shard: u64,
        scale_path: Option<&Path>,
        row_stride: usize,
        slots: usize,
    ) -> Result<Self> {
        if bases.is_empty() || rows_per_shard == 0 {
            bail!(
                "NgramRowCache: segmented table needs shards and rows \
                 (shards={}, rows_per_shard={rows_per_shard})",
                bases.len()
            );
        }
        if paths.is_empty() {
            bail!("NgramRowCache: segmented table needs at least one backing file");
        }
        if file_of.len() != bases.len() {
            bail!(
                "NgramRowCache: {} shard bases but {} file assignments",
                bases.len(),
                file_of.len()
            );
        }
        if let Some(bad) = file_of.iter().find(|f| **f as usize >= paths.len()) {
            bail!(
                "NgramRowCache: shard names file {bad} but only {} were given",
                paths.len()
            );
        }
        let rows_total = bases.len() as u64 * rows_per_shard;
        let mut c = Self::open_at(paths[0], 0, scale_path, rows_total, row_stride, slots)?;
        for p in &paths[1..] {
            c.files.push(open_direct(p)?);
        }
        c.segments = Some(Segments {
            bases,
            file_of,
            rows_per: rows_per_shard,
        });
        Ok(c)
    }

    /// Device VA of the cache's row table — the `embed_table` argument of the
    /// gather kernels, which then index it by SLOT.
    pub fn table_dev_va(&self) -> Result<u64> {
        self.arena.slot_dev_va(0, 0)
    }

    /// Device VA of the `[slots]` f32 scale array (FP8 tables only).
    pub fn scale_dev_va(&self) -> Result<Option<u64>> {
        match &self.scales {
            Some(s) => Ok(Some(s.arena.slot_dev_va(0, 0)?)),
            None => Ok(None),
        }
    }

    pub fn stats(&self) -> (u64, u64, u64) {
        (self.hits, self.misses, self.evictions)
    }

    /// Resolve `row_ids` to slot indices, faulting misses in from NVMe.
    ///
    /// Every returned slot is PINNED for the caller's batch: the gather runs
    /// after this returns, so a later resolve in the same batch must not
    /// evict a row the kernel is about to read. Call [`Self::end_batch`] once
    /// the gather has been issued.
    ///
    /// An ERROR aborts the whole batch: no gather will consume the slots, so
    /// every pin is released and this call's map inserts are rolled back
    /// before returning. Pins used to survive a failed resolve — each
    /// all-slots-pinned refusal then leaked its pins permanently, and after a
    /// few such failures every slot was pinned forever: the server answered
    /// nothing until restart (observed 2026-08-28 at chunk 8192 x 16 heads
    /// against 65,536 slots). The victim-exhaustion path also left map
    /// entries pointing at slots whose rows were never faulted in — a later
    /// hit on such an entry would have gathered garbage.
    pub fn resolve(&mut self, row_ids: &[u64], out_slots: &mut Vec<u32>) -> Result<()> {
        out_slots.clear();
        out_slots.reserve(row_ids.len());
        // Phase 1 — bookkeeping only: pin hits, assign a victim slot to every
        // miss (a repeated missing id hits the map on its second occurrence,
        // so each unique row faults once). No I/O under this loop.
        let mut jobs: Vec<crate::ngram_cache_fault::FaultJob> = Vec::new();
        let mut phase1_err: Option<anyhow::Error> = None;
        for &id in row_ids {
            if id >= self.rows_total {
                phase1_err = Some(anyhow::anyhow!(
                    "NgramRowCache: row id {id} >= table rows {} (hash/table mismatch)",
                    self.rows_total
                ));
                break;
            }
            let slot = match self.map.get(&id) {
                Some(&s) => {
                    self.hits += 1;
                    self.refbit[s as usize] = true;
                    self.pinned[s as usize] = true;
                    s
                }
                None => {
                    self.misses += 1;
                    let s = match self.victim() {
                        Ok(s) => s,
                        Err(e) => {
                            phase1_err = Some(e);
                            break;
                        }
                    };
                    self.map.insert(id, s);
                    self.slot_row[s as usize] = id;
                    self.refbit[s as usize] = true;
                    self.pinned[s as usize] = true;
                    match self.fault_job(id, s) {
                        Ok(j) => jobs.push(j),
                        Err(e) => {
                            phase1_err = Some(e);
                            break;
                        }
                    }
                    s
                }
            };
            out_slots.push(slot);
        }
        if let Some(e) = phase1_err {
            self.abort_batch(&jobs);
            out_slots.clear();
            return Err(e);
        }
        // Phase 2 — fault every miss in, parallel past a few (the serial
        // QD=1 pread-per-miss loop was the diverse-prefill stall).
        if !jobs.is_empty() {
            let r = crate::ngram_cache_fault::fault_all(
                &jobs,
                &self.files,
                self.scales.as_ref().map(|sc| &sc.file),
                self.row_stride,
                &mut self.bounce,
            );
            if let Err(e) = r {
                self.abort_batch(&jobs);
                out_slots.clear();
                return Err(e);
            }
        }
        Ok(())
    }

    /// Error-path rollback: this call's map inserts describe slots whose rows
    /// were never (fully) faulted in — remove them so a later hit cannot
    /// gather garbage — and release EVERY pin, because the aborted batch's
    /// gather will never run and pins have no other release point.
    fn abort_batch(&mut self, jobs: &[crate::ngram_cache_fault::FaultJob]) {
        for j in jobs {
            self.map.remove(&j.row_id);
            self.slot_row[j.slot as usize] = u64::MAX;
            self.refbit[j.slot as usize] = false;
        }
        self.end_batch();
    }

    /// Resolve one miss to byte offsets + destination addresses — the
    /// bookkeeping-free half of the old `fetch_into`, consumed by
    /// [`crate::ngram_cache_fault::fault_all`].
    fn fault_job(&self, id: u64, slot: u32) -> Result<crate::ngram_cache_fault::FaultJob> {
        let (file_idx, byte) = self.row_byte(id);
        let block_off = byte - (byte % BLOCK as u64);
        let within = (byte - block_off) as usize;
        let nblocks = crate::ngram_cache_fault::nblocks_for(within, self.row_stride);
        // SAFETY: address arithmetic only; the fault worker writes the
        // disjoint `[dst, dst+row_stride)` region while the arena is live.
        let dst = unsafe {
            self.arena
                .slot_host_ptr(0, 0)?
                .add(slot as usize * self.row_stride)
        } as usize;
        let scale = match &self.scales {
            Some(sc) => {
                let sbyte = id * 4;
                let sblock = sbyte - (sbyte % BLOCK as u64);
                let swithin = (sbyte - sblock) as usize;
                // SAFETY: as above, 4-byte disjoint region.
                let sdst = unsafe { sc.arena.slot_host_ptr(0, 0)?.add(slot as usize * 4) };
                Some((sblock, swithin, sdst as usize))
            }
            None => None,
        };
        Ok(crate::ngram_cache_fault::FaultJob {
            row_id: id,
            slot,
            file_idx,
            block_off,
            within,
            nblocks,
            dst,
            scale,
        })
    }

    /// Release the batch's pins (call after the gather kernels are issued).
    pub fn end_batch(&mut self) {
        for p in &mut self.pinned {
            *p = false;
        }
    }

    /// CLOCK second-chance victim among the unpinned slots.
    fn victim(&mut self) -> Result<u32> {
        for _ in 0..(self.slots * 2) {
            let s = self.hand;
            self.hand = (self.hand + 1) % self.slots;
            if self.pinned[s] {
                continue;
            }
            if self.refbit[s] {
                self.refbit[s] = false;
                continue;
            }
            if self.slot_row[s] != u64::MAX {
                let old = self.slot_row[s];
                self.map.remove(&old);
                self.evictions += 1;
            }
            return Ok(s as u32);
        }
        bail!(
            "NgramRowCache: every one of {} slots is pinned by the batch in flight — \
             raise the cache size or lower max-prefill-tokens",
            self.slots
        )
    }

    /// Which backing file row `id` lives in, and its byte offset there.
    fn row_byte(&self, id: u64) -> (u16, u64) {
        match &self.segments {
            None => (0, self.base_offset + id * self.row_stride as u64),
            Some(seg) => {
                let shard = (id / seg.rows_per) as usize;
                let local = id % seg.rows_per;
                (
                    seg.file_of[shard],
                    seg.bases[shard] + local * self.row_stride as u64,
                )
            }
        }
    }
}

#[cfg(unix)]
fn open_direct(path: &Path) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .with_context(|| format!("NgramRowCache: open {} (O_DIRECT)", path.display()))
}

#[cfg(not(unix))]
fn open_direct(path: &Path) -> Result<File> {
    File::open(path).with_context(|| format!("NgramRowCache: open {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_scratch_is_4k_aligned_and_two_blocks() {
        let mut b = AlignedBlock::new();
        let s = b.blocks(2);
        assert_eq!(s.len(), BLOCK * 2);
        assert_eq!(s.as_ptr() as usize % BLOCK, 0);
    }

    #[test]
    fn row_wider_than_a_block_is_refused() {
        // A row larger than one block could span three with an unaligned base.
        let msg = match NgramRowCache::open(Path::new("/nonexistent"), None, 10, BLOCK + 8, 4) {
            Ok(_) => panic!("expected refusal for oversize row_stride"),
            Err(e) => format!("{e:#}"),
        };
        assert!(msg.contains("O_DIRECT block"), "{msg}");
    }

    /// Regression: a resolve that FAILS (more unique rows than slots) must
    /// not poison the cache. Before the abort_batch rollback, each such
    /// failure leaked its pins permanently — after a few failures every slot
    /// was pinned and the server answered nothing until restart (2026-08-28,
    /// chunk 8192 x 16 heads vs 65,536 slots) — and the failed batch's map
    /// entries pointed at slots whose rows were never faulted in.
    /// GPU test (pinned arena): run with `-- --ignored`.
    #[test]
    #[ignore]
    fn failed_resolve_releases_pins_and_rolls_back_map() {
        // Pinned arena needs a live CUDA context (same pattern as the
        // expert_arena GPU tests).
        let _ctx = crate::cuda_min::CudaCtx::new(0).unwrap();
        let stride = 256usize;
        let rows = 64u64;
        let dir = std::env::temp_dir().join(format!("ngram_cache_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("table.bin");
        let mut data = vec![0u8; rows as usize * stride];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        std::fs::write(&path, &data).unwrap();

        let mut cache = NgramRowCache::open(&path, None, rows, stride, 8).unwrap();
        let mut slots = Vec::new();

        // 9 unique rows > 8 slots: must refuse...
        let too_many: Vec<u64> = (0..9).collect();
        assert!(cache.resolve(&too_many, &mut slots).is_err());

        // ...and afterwards a batch that fits must succeed — including ids
        // from the failed batch (their rolled-back map entries must fault in
        // cleanly, not "hit" a garbage slot).
        let ok: Vec<u64> = (0..8).collect();
        cache.resolve(&ok, &mut slots).expect("cache poisoned by failed resolve");
        assert_eq!(slots.len(), 8);
        cache.end_batch();

        // Repeatedly failing must not accumulate pins (the production brick).
        for _ in 0..5 {
            assert!(cache.resolve(&too_many, &mut slots).is_err());
        }
        cache.resolve(&[40, 41], &mut slots).expect("pins leaked across failed resolves");
        cache.end_batch();

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The seam arithmetic: with a base offset that is only 8-byte aligned
    /// (what a safetensors shard gives), rows land at every phase relative to
    /// the 4 KiB block, and the covering span must stay within two blocks.
    #[test]
    fn straddling_rows_are_covered_by_two_blocks() {
        for base in [0u64, 8, 1234568, 4095, 4097] {
            for stride in [256usize, 512, 4096] {
                for id in [0u64, 1, 7, 8, 1023] {
                    let byte = base + id * stride as u64;
                    let block_off = byte - (byte % BLOCK as u64);
                    let within = (byte - block_off) as usize;
                    let n = if within + stride > BLOCK { 2 } else { 1 };
                    assert!(
                        within + stride <= n * BLOCK,
                        "base={base} stride={stride} id={id} within={within} n={n}"
                    );
                }
            }
        }
    }
}
