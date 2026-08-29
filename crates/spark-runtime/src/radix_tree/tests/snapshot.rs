// SPDX-License-Identifier: AGPL-3.0-only

//! Snapshot-side tests: intermediate snapshots and partial-suffix matching.
//! The standalone `SsmSnapshotIndex` LRU/session/overwrite behaviours live in
//! `snapshot_index.rs` (mounted as `radix_tree::snapshot`'s unit tests).

use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

use super::super::hash_token_prefix;
use super::super::snapshot::SsmSnapshotIndex;

#[test]
fn test_insert_without_snapshot() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();

    tree.insert(&tokens, &[10], &[], 16, 0, 0);
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.ssm_snapshot, None);
    assert_eq!(m.ssm_snapshot_tokens, 0);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_intermediate_snapshot_on_partial_match() {
    let tree = RadixTree::new();

    // Insert 4-block sequence
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert(&tokens, &[10, 20, 30, 40], &[], 16, 0, 0);

    // Attach intermediate snapshot at block 2 (token 32)
    let tokens_at_2: Vec<u32> = (0..32).collect();
    tree.insert_intermediate_snapshot(&tokens_at_2, &[10, 20], &[], 16, 50, 0, 0, 0);

    // Lookup all 4 blocks — should return intermediate snapshot at block 2
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.matched_tokens, 64);
    assert_eq!(m.ssm_snapshot, Some(50));
    assert_eq!(m.ssm_snapshot_tokens, 32);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_intermediate_snapshot_deepest_wins() {
    let tree = RadixTree::new();

    // Insert 4-block sequence with leaf snapshot
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert_with_snapshot(&tokens, &[10, 20, 30, 40], &[], 16, 99, 0, 0, 0);

    // Attach intermediate snapshot at block 2 (token 32)
    let tokens_at_2: Vec<u32> = (0..32).collect();
    tree.insert_intermediate_snapshot(&tokens_at_2, &[10, 20], &[], 16, 50, 0, 0, 0);

    // Lookup all 4 blocks — leaf snapshot (deeper) wins
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.matched_tokens, 64);
    assert_eq!(m.ssm_snapshot, Some(99));
    assert_eq!(m.ssm_snapshot_tokens, 64);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_intermediate_snapshot_partial_prefix_hit() {
    let tree = RadixTree::new();

    // Insert 4-block sequence
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert(&tokens, &[10, 20, 30, 40], &[], 16, 0, 0);

    // Attach intermediate snapshot at block 2 (token 32)
    let tokens_at_2: Vec<u32> = (0..32).collect();
    tree.insert_intermediate_snapshot(&tokens_at_2, &[10, 20], &[], 16, 50, 0, 0, 0);

    // New request shares first 48 tokens, diverges at block 4
    let mut tokens_new: Vec<u32> = (0..48).collect();
    tokens_new.extend(200..216);
    let m = tree.lookup(&tokens_new, 16, 0, 0);
    // Matches 3 blocks (48 tokens), intermediate snapshot at block 2
    assert_eq!(m.matched_tokens, 48);
    assert_eq!(m.ssm_snapshot, Some(50));
    assert_eq!(m.ssm_snapshot_tokens, 32);
    tree.release(&tokens_new, 16, 0);
}

#[test]
fn test_intermediate_snapshot_survives_tree_eviction() {
    let tree = RadixTree::new();

    // Insert 2-block sequence with intermediate snapshot on block 1
    let tokens: Vec<u32> = (0..32).collect();
    tree.insert(&tokens, &[10, 20], &[], 16, 0, 0);
    tree.release(&tokens, 16, 0); // inserting seq exits → nodes evictable

    let tokens_at_1: Vec<u32> = (0..16).collect();
    tree.insert_intermediate_snapshot(&tokens_at_1, &[10], &[], 16, 50, 0, 0, 0);

    // Evict both tree nodes — snapshot survives in index
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![20]);
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![10]);

    // Snapshot still in index (decoupled from tree)
    assert_eq!(tree.snapshot_count(), 1);
    let snap = tree.evict_snapshot_lru();
    assert_eq!(snap, Some(50));
}

// ── Partial suffix tests ──

#[test]
fn test_partial_suffix_insert_and_lookup() {
    let tree = RadixTree::new();
    // 20 tokens = 1 full block (16) + 4 partial
    let tokens: Vec<u32> = (0..20).collect();
    let block_table = vec![10, 20]; // block for full + block for partial

    tree.insert(&tokens, &block_table, &[], 16, 0, 0);
    let m = tree.lookup(&tokens, 16, 0, 0);

    // Should match all 20 tokens (16 full + 4 partial)
    assert_eq!(m.matched_tokens, 20);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_partial_suffix_no_match_different_suffix() {
    let tree = RadixTree::new();
    // Insert 20 tokens
    let tokens_a: Vec<u32> = (0..20).collect();
    tree.insert(&tokens_a, &[10, 20], &[], 16, 0, 0);

    // Lookup 20 tokens with different suffix (same first 16, different last 4)
    let mut tokens_b: Vec<u32> = (0..16).collect();
    tokens_b.extend(100..104);
    let m = tree.lookup(&tokens_b, 16, 0, 0);

    // Should match only 16 full-block tokens (partial suffix doesn't match)
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&tokens_b, 16, 0);
}

#[test]
fn test_partial_suffix_not_matched_for_full_block_request() {
    let tree = RadixTree::new();
    // Insert 20 tokens (1 full + 4 partial)
    let tokens: Vec<u32> = (0..20).collect();
    tree.insert(&tokens, &[10, 20], &[], 16, 0, 0);

    // Lookup 32 tokens — 2 full blocks in request. Partial suffix is 4 tokens
    // but remainder is 0 (32 % 16 == 0), so partial check is skipped.
    let tokens_32: Vec<u32> = (0..32).collect();
    let m = tree.lookup(&tokens_32, 16, 0, 0);

    // Only first full block matches (second block [16..32] has no matching tree node)
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&tokens_32, 16, 0);
}

#[test]
fn test_partial_suffix_eviction_frees_both_blocks() {
    let tree = RadixTree::new();
    // Insert 20 tokens (1 full block + 4 partial) + release inserting seq
    let tokens: Vec<u32> = (0..20).collect();
    tree.insert(&tokens, &[10, 20], &[], 16, 0, 0);
    tree.release(&tokens, 16, 0);

    // Evict 1 — should free block 10 (full) AND block 20 (partial suffix)
    let evicted = tree.evict(1);
    // Evicting the leaf node also frees its partial suffix block
    assert!(evicted.physical.contains(&10));
    assert!(evicted.physical.contains(&20));
}

#[test]
#[ignore = "tests removed behavior — partial-suffix clearing was replaced \
            with partial-block-matching during the radix-tree refactor; \
            assertions need rewriting against the new lookup semantics"]
fn test_partial_suffix_cleared_when_extended() {
    let tree = RadixTree::new();
    // Insert 20 tokens (1 full + 4 partial)
    let tokens_20: Vec<u32> = (0..20).collect();
    tree.insert(&tokens_20, &[10, 20], &[], 16, 0, 0);

    // Insert 32 tokens (2 full blocks, extends past partial)
    let tokens_32: Vec<u32> = (0..32).collect();
    tree.insert(&tokens_32, &[10, 30], &[], 16, 0, 0);

    // Lookup 20 tokens — partial suffix was cleared by the 32-token insert
    let m = tree.lookup(&tokens_20, 16, 0, 0);
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&tokens_20, 16, 0);

    // Lookup 32 tokens — full match
    let m = tree.lookup(&tokens_32, 16, 0, 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.matched_blocks, vec![10, 30]);
    tree.release(&tokens_32, 16, 0);
}

#[test]
fn test_partial_suffix_multi_block_prefix() {
    let tree = RadixTree::new();
    // 396 tokens = 24 full blocks + 12 partial
    let tokens: Vec<u32> = (0..396).collect();
    let block_table: Vec<u32> = (0..25).collect();
    // block_table[24] = partial block

    tree.insert(&tokens, &block_table, &[], 16, 0, 0);
    let m = tree.lookup(&tokens, 16, 0, 0);

    assert_eq!(m.matched_tokens, 396);
    assert_eq!(m.matched_blocks.len(), 25);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_partial_suffix_prefix_match_shorter_lookup() {
    let tree = RadixTree::new();
    // Insert 31 tokens (1 full block + 15 partial) — simulates prompt+generation
    let tokens_31: Vec<u32> = (0..31).collect();
    tree.insert(&tokens_31, &[10, 20], &[], 16, 0, 0);

    // Lookup 22 tokens (1 full block + 6 partial) — simulates repeat of prompt only
    let tokens_22: Vec<u32> = (0..22).collect();
    let m = tree.lookup(&tokens_22, 16, 0, 0);

    // Partial suffix [16..31] starts with [16..22], so prefix match succeeds
    assert_eq!(m.matched_tokens, 22);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens_22, 16, 0);
}

#[test]
fn test_sub_block_match_via_child_key_prefix() {
    let tree = RadixTree::new();
    // Insert 35 tokens (2 full blocks + 3 partial) — prompt + generation
    let tokens_35: Vec<u32> = (0..35).collect();
    tree.insert(&tokens_35, &[10, 20, 30], &[], 16, 0, 0);

    // Lookup 22 tokens (1 full block + 6 remaining) — same prompt
    let tokens_22: Vec<u32> = (0..22).collect();
    let m = tree.lookup(&tokens_22, 16, 0, 0);

    // Block 0 (0-15) matched as full block.
    // Remaining 6 tokens (16-21) are a prefix of block 1's key (16-31).
    // Sub-block matching should include block 1.
    assert_eq!(m.matched_tokens, 22);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens_22, 16, 0);
}

#[test]
fn test_partial_suffix_sub_block_only() {
    let tree = RadixTree::new();
    // Only 10 tokens — no full blocks, partial suffix not stored (no parent)
    let tokens: Vec<u32> = (0..10).collect();
    tree.insert(&tokens, &[42], &[], 16, 0, 0);

    // No full blocks → nothing cached or matched
    assert_eq!(tree.stats(), (0, 0));
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.matched_tokens, 0);
}

// ── Task #24: adapter-correct SSM snapshots + base hash byte-identity ──

/// `hash_token_prefix(_, _, 0)` (base sentinel) must reduce EXACTLY to the
/// pre-#24 token-only FNV-1a value, so base prefix-cache/snapshot hit rates are
/// unchanged. A non-zero adapter_id must change the hash.
#[test]
fn test_hash_token_prefix_base_byte_identical() {
    let tokens: Vec<u32> = vec![7, 42, 1000, 65535, 3, 0, 128];
    // Recompute the exact pre-#24 formula inline.
    let mut expected: u64 = 0xcbf29ce484222325;
    for &t in &tokens {
        expected ^= t as u64;
        expected = expected.wrapping_mul(0x100000001b3);
    }
    assert_eq!(
        hash_token_prefix(&tokens, tokens.len(), 0),
        expected,
        "base (adapter_id=0) hash must be byte-identical to the pre-#24 value"
    );
    // Any non-zero adapter partitions the key.
    assert_ne!(
        hash_token_prefix(&tokens, tokens.len(), 0),
        hash_token_prefix(&tokens, tokens.len(), 99),
    );
    assert_ne!(
        hash_token_prefix(&tokens, tokens.len(), 7),
        hash_token_prefix(&tokens, tokens.len(), 9),
    );
}

/// The SSM snapshot index must isolate by adapter: a snapshot registered under
/// adapter A's prefix hash is not found by an adapter-B lookup, but is by an
/// adapter-A lookup.
#[test]
fn test_snapshot_index_adapter_isolation() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..16).collect();
    const A: u64 = 0xAA;
    const B: u64 = 0xBB;

    // Register under adapter A (the tree computes prefix_hash with A folded in).
    let ph_a = hash_token_prefix(&tokens, 16, A);
    idx.insert(ph_a, 42, 0, 16);

    // Adapter B lookup recomputes with B → different hash → miss.
    assert_eq!(idx.lookup(&tokens, 16, 0, B), None);
    // Adapter A lookup → hit.
    assert_eq!(idx.lookup(&tokens, 16, 0, A), Some((42, 16)));
    // Base lookup → miss (base hash != A hash).
    assert_eq!(idx.lookup(&tokens, 16, 0, 0), None);
}

/// End-to-end through the tree API: an SSM snapshot saved under adapter A is
/// not restored for an adapter-B request, but is for an adapter-A request.
#[test]
fn test_ssm_snapshot_adapter_isolation_via_tree() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    const A: u64 = 0x55;
    const B: u64 = 0x66;

    tree.insert_with_snapshot(&tokens, &[10, 20], &[], 16, 42, 0, 0, A);
    tree.release(&tokens, 16, A);

    // Adapter B: KV misses AND no snapshot restore.
    let m_b = tree.lookup(&tokens, 16, 0, B);
    assert!(m_b.is_empty());
    assert_eq!(m_b.ssm_snapshot, None);

    // Adapter A: KV hit + snapshot restored.
    let m_a = tree.lookup(&tokens, 16, 0, A);
    assert_eq!(m_a.matched_tokens, 32);
    assert_eq!(m_a.ssm_snapshot, Some(42));
    tree.release(&tokens, 16, A);
}

#[test]
fn test_decode_ckpt_ring_supersedes_shallowest_and_spares_prefill_anchor() {
    let tree = RadixTree::new();
    const SESSION: u64 = 0xABC;

    // Prefill lays 8 blocks and a prompt-boundary intermediate anchor at
    // token 32 (snapshot 7) — the entry the ring must NOT displace.
    let tokens: Vec<u32> = (0..128).collect();
    tree.insert(&tokens, &[10, 20, 30, 40, 50, 60, 70, 80], &[], 16, 0, 0);
    let anchor: Vec<u32> = (0..32).collect();
    tree.insert_intermediate_snapshot(&anchor, &[10, 20], &[], 16, 7, SESSION, 0, 0);

    // Decode fires checkpoints at tokens 64 / 80 / 96 / 112 (ids 100..104).
    // With ATLAS_DECODE_CKPT_KEEP unset the ring keeps 2: each insert past
    // the second displaces the SHALLOWEST decode ckpt, never the anchor.
    let mut displaced_all = Vec::new();
    for (i, end) in [64usize, 80, 96, 112].iter().enumerate() {
        let t: Vec<u32> = (0..*end as u32).collect();
        displaced_all.extend(tree.insert_decode_ckpt_snapshot(&t, 100 + i, SESSION, 0));
    }
    // 4 inserted, ring of 2 → the two shallowest (ids 100, 101) came back.
    displaced_all.sort_unstable();
    assert_eq!(displaced_all, vec![100, 101]);

    // Deepest decode ckpt wins a full-prefix lookup…
    let m = tree.lookup(&tokens, 16, SESSION, 0);
    assert_eq!(m.matched_tokens, 128);
    assert_eq!(m.ssm_snapshot, Some(103));
    assert_eq!(m.ssm_snapshot_tokens, 112);
    tree.release(&tokens, 16, 0);

    // …and the prompt-boundary anchor still serves a seam-diverged warm turn
    // (match capped below every surviving decode ckpt).
    let mut diverged: Vec<u32> = (0..48).collect();
    diverged[33] = 9999; // diverges inside block 3 → radix match stops at 32
    let m2 = tree.lookup(&diverged, 16, SESSION, 0);
    assert_eq!(m2.matched_tokens, 32);
    assert_eq!(
        m2.ssm_snapshot,
        Some(7),
        "prefill anchor must survive the decode-ckpt churn"
    );
    assert_eq!(m2.ssm_snapshot_tokens, 32);
    tree.release(&diverged[..32].to_vec(), 16, 0);
}

#[test]
fn test_decode_ckpt_ring_is_per_session() {
    let tree = RadixTree::new();
    let t64: Vec<u32> = (0..64).collect();
    let t80: Vec<u32> = (0..80).collect();
    let t96: Vec<u32> = (0..96).collect();
    tree.insert(&t96, &[10, 20, 30, 40, 50, 60], &[], 16, 0, 0);

    // Session A holds two ckpts; session B's inserts must not sweep them.
    assert!(tree.insert_decode_ckpt_snapshot(&t64, 1, 0xA, 0).is_empty());
    assert!(tree.insert_decode_ckpt_snapshot(&t80, 2, 0xA, 0).is_empty());
    let u64_: Vec<u32> = (1000..1064).collect();
    let u80_: Vec<u32> = (1000..1080).collect();
    let u96_: Vec<u32> = (1000..1096).collect();
    assert!(
        tree.insert_decode_ckpt_snapshot(&u64_, 3, 0xB, 0)
            .is_empty()
    );
    assert!(
        tree.insert_decode_ckpt_snapshot(&u80_, 4, 0xB, 0)
            .is_empty()
    );
    // Third B ckpt sweeps B's shallowest (id 3), not anything of A.
    assert_eq!(tree.insert_decode_ckpt_snapshot(&u96_, 5, 0xB, 0), vec![3]);
    let m = tree.lookup(&t96, 16, 0xA, 0);
    assert_eq!(m.ssm_snapshot, Some(2), "session A's ring untouched");
    tree.release(&t96, 16, 0);
}

#[test]
fn test_intermediate_ring_supersedes_shallowest_and_spares_the_boundary_anchor() {
    let tree = RadixTree::new();
    const SESSION: u64 = 0xBEEF;

    // A long chunked prefill: the tree gets the whole prompt, then the
    // per-chunk intermediate checkpoints fire at 32 / 48 / 64 / 80, and the
    // turn's finish-leaf lands the boundary anchor (id 9) at token 16.
    let tokens: Vec<u32> = (0..128).collect();
    tree.insert(&tokens, &[10, 20, 30, 40, 50, 60, 70, 80], &[], 16, 0, 0);
    let anchor: Vec<u32> = (0..16).collect();
    // `insert_with_snapshot` is what finalize_last does — a durable anchor,
    // not an intermediate.
    tree.insert_with_snapshot(&anchor, &[10], &[], 16, 9, 0, 0, 0);

    let mut displaced = Vec::new();
    for (i, end) in [32usize, 48, 64, 80].iter().enumerate() {
        let t: Vec<u32> = (0..*end as u32).collect();
        displaced.extend(tree.insert_intermediate_snapshot(&t, &[10, 20], &[], 16, 20 + i, SESSION, 0, 0));
    }
    // Ring of 2 (ATLAS_SSM_INTERMEDIATE_KEEP default): the two shallowest
    // intermediates came back, and the anchor was never a candidate.
    displaced.sort_unstable();
    assert_eq!(displaced, vec![20, 21]);

    // The deepest surviving intermediate serves a resume inside this turn…
    let t80: Vec<u32> = (0..80).collect();
    let m = tree.lookup(&t80, 16, SESSION, 0);
    assert_eq!(m.ssm_snapshot, Some(23));
    tree.release(&t80, 16, 0);

    // …and the boundary anchor still serves the next turn's shallower match.
    let mut diverged: Vec<u32> = (0..32).collect();
    diverged[20] = 7777; // diverges in block 1 → radix match stops at 16
    let m2 = tree.lookup(&diverged, 16, SESSION, 0);
    assert_eq!(m2.matched_tokens, 16);
    assert_eq!(
        m2.ssm_snapshot,
        Some(9),
        "the boundary anchor must survive the intermediate churn"
    );
    tree.release(&diverged[..16].to_vec(), 16, 0);
}

#[test]
fn test_intermediate_ring_is_per_session() {
    let tree = RadixTree::new();
    let t32: Vec<u32> = (0..32).collect();
    let t48: Vec<u32> = (0..48).collect();
    tree.insert(&t48, &[10, 20, 30], &[], 16, 0, 0);

    // Session A fills its ring; session B's prefill must not sweep it — the
    // 2026-08-29 failure was exactly four sessions evicting each other.
    assert!(
        tree.insert_intermediate_snapshot(&t32, &[10, 20], &[], 16, 1, 0xA, 0, 0)
            .is_empty()
    );
    assert!(
        tree.insert_intermediate_snapshot(&t48, &[10, 20, 30], &[], 16, 2, 0xA, 0, 0)
            .is_empty()
    );
    let u32_: Vec<u32> = (2000..2032).collect();
    let u48_: Vec<u32> = (2000..2048).collect();
    let u64_: Vec<u32> = (2000..2064).collect();
    assert!(
        tree.insert_intermediate_snapshot(&u32_, &[10], &[], 16, 3, 0xB, 0, 0)
            .is_empty()
    );
    assert!(
        tree.insert_intermediate_snapshot(&u48_, &[10], &[], 16, 4, 0xB, 0, 0)
            .is_empty()
    );
    assert_eq!(
        tree.insert_intermediate_snapshot(&u64_, &[10], &[], 16, 5, 0xB, 0, 0),
        vec![3],
        "B's third sweeps B's shallowest, nothing of A"
    );
    let m = tree.lookup(&t48, 16, 0xA, 0);
    assert_eq!(m.ssm_snapshot, Some(2), "session A's ring untouched");
    tree.release(&t48, 16, 0);
}

#[test]
fn test_finish_leaf_over_an_intermediate_prefix_promotes_it_to_an_anchor() {
    // A prompt whose last chunk ends exactly on a checkpoint boundary saves an
    // intermediate and then the finish-leaf at the SAME prefix. The leaf must
    // clear the intermediate flag, or the next turn's ring sweep would treat
    // the turn's own restore point as disposable.
    let tree = RadixTree::new();
    const SESSION: u64 = 0xF00D;
    let t32: Vec<u32> = (0..32).collect();
    tree.insert(&t32, &[10, 20], &[], 16, 0, 0);
    assert!(
        tree.insert_intermediate_snapshot(&t32, &[10, 20], &[], 16, 1, SESSION, 0, 0)
            .is_empty()
    );
    // finalize_last's insert over the same prefix hands back slot 1.
    let (displaced, _) = tree.insert_with_snapshot(&t32, &[10, 20], &[], 16, 2, SESSION, 0, 0);
    assert_eq!(displaced, Some(1));

    // Two later intermediates would have swept a 3-deep ring; the promoted
    // anchor is not in it, so nothing is displaced.
    for (i, end) in [48usize, 64].iter().enumerate() {
        let t: Vec<u32> = (0..*end as u32).collect();
        tree.insert(&t, &[10, 20, 30, 40], &[], 16, 0, 0);
        assert!(
            tree.insert_intermediate_snapshot(&t, &[10, 20], &[], 16, 30 + i, SESSION, 0, 0)
                .is_empty(),
            "the promoted anchor must not be swept by the intermediate ring"
        );
    }
    let m = tree.lookup(&t32, 16, SESSION, 0);
    assert_eq!(m.ssm_snapshot, Some(2));
    tree.release(&t32, 16, 0);
}

#[test]
fn test_turn_anchor_promotion_survives_the_next_turns_intermediate_ring() {
    // The exact 2026-08-29 replay failure. Turn A's prompt ends at 15051, so
    // its intermediates land at 8196 and 15024 and the next turn's radix match
    // stops at 15024 (block-floored). Turn B's intermediates (8196 / 16388 /
    // 19232) are all DEEPER, so an un-promoted 15024 is swept by B's own ring
    // and turn B re-prefills the whole conversation.
    let tree = RadixTree::new();
    const S: u64 = 0xC0FFEE;
    let all: Vec<u32> = (0..20000).collect();
    let blocks: Vec<u32> = (0..1250).collect();
    tree.insert(&all, &blocks, &[], 16, 0, 0);

    let at = |n: usize| -> Vec<u32> { (0..n as u32).collect() };
    // Turn A.
    tree.insert_intermediate_snapshot(&at(8196), &blocks[..512], &[], 16, 1, S, 0, 0);
    tree.insert_intermediate_snapshot(&at(15024), &blocks[..939], &[], 16, 2, S, 0, 0);
    // finalize: leaf at 15051, then promote the deepest intermediate (15024).
    tree.insert_with_snapshot(&at(15051), &blocks[..941], &[], 16, 3, S, 0, 0);
    assert!(
        tree.promote_turn_anchor(S).is_empty(),
        "first promotion displaces nothing"
    );

    // Turn B: three deeper intermediates. A ring of 2 would have evicted 15024.
    let mut displaced = Vec::new();
    for (i, n) in [8196usize, 16388, 19232].iter().enumerate() {
        displaced.extend(tree.insert_intermediate_snapshot(
            &at(*n),
            &blocks[..*n / 16],
            &[],
            16,
            10 + i,
            S,
            0,
            0,
        ));
    }

    // The promoted anchor is still there and still serves the seam.
    let m = tree.lookup(&at(15024), 16, S, 0);
    assert_eq!(m.matched_tokens, 15024);
    assert_eq!(
        m.ssm_snapshot,
        Some(2),
        "turn A's block-floored anchor must survive turn B's intermediates; \
         displaced={displaced:?}"
    );
    tree.release(&at(15024), 16, 0);

    // Turn B's own promotion supersedes turn A's — one anchor per session.
    tree.insert_with_snapshot(&at(19251), &blocks[..1203], &[], 16, 20, S, 0, 0);
    assert_eq!(
        tree.promote_turn_anchor(S),
        vec![2],
        "turn B's anchor replaces turn A's"
    );
}

#[test]
fn test_promote_turn_anchor_is_per_session_and_noop_without_intermediates() {
    let tree = RadixTree::new();
    let at = |n: usize| -> Vec<u32> { (0..n as u32).collect() };
    let blocks: Vec<u32> = (0..1250).collect();
    tree.insert(&at(20000), &blocks, &[], 16, 0, 0);

    // No intermediates for this session yet — nothing to promote.
    assert!(tree.promote_turn_anchor(0xAA).is_empty());

    tree.insert_intermediate_snapshot(&at(4096), &blocks[..256], &[], 16, 1, 0xAA, 0, 0);
    tree.insert_intermediate_snapshot(&at(8192), &blocks[..512], &[], 16, 2, 0xBB, 0, 0);
    // Promoting A must not touch B's intermediate…
    assert!(tree.promote_turn_anchor(0xAA).is_empty());
    // …and B's promotion picks B's own, not the deeper-or-shallower of A's.
    assert!(tree.promote_turn_anchor(0xBB).is_empty());
    let m = tree.lookup(&at(8192), 16, 0xBB, 0);
    assert_eq!(m.ssm_snapshot, Some(2));
    tree.release(&at(8192), 16, 0);
}
