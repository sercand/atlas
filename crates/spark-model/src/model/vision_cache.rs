// SPDX-License-Identifier: AGPL-3.0-only

//! Image-hash-aware prefix caching (2026-08-28).
//!
//! The prefix cache and the SSM snapshot index address KV/state by the raw
//! token stream. Vision prompts break that addressing: every image renders as
//! a run of identical `<|image_pad|>` tokens, so two prompts that differ only
//! in image CONTENT are token-identical — a cache match across them would
//! silently reuse the wrong image's KV. The historical fix was a blanket veto
//! (`tokens_have_vision_pad` → no lookup, no insert), which costs a full
//! re-prefill on EVERY turn of a conversation once a single screenshot enters
//! it (observed 2026-08-28: 18-21 s TTFT per turn after one pasted image).
//!
//! This module makes the token stream content-addressed instead: for cache
//! and snapshot purposes ONLY, each pad token is replaced by a **virtual id**
//! derived from (that image's content hash, position within its pad run).
//! Identical text + identical images → identical virtual streams → a safe
//! match; a different image diverges at its first pad token. Virtual ids have
//! bit 31 set, which no real vocab id reaches (Qwen3.8 vocab ≈ 249k ≪ 2^31),
//! so a virtual stream can never alias a real token sequence. The REAL token
//! stream is untouched — embedding lookup, the vision splice, MRoPE and PLE
//! all still see the genuine pad ids.
//!
//! Soundness of reuse: a matched prefix implies identical text tokens AND
//! identical per-image content hashes at identical positions, so the vision
//! embeddings spliced at insert time equal the ones this request would
//! compute — the cached KV (and any SSM snapshot keyed on the same virtual
//! stream) is byte-valid. Restore-side hazards (a resume point inside a pad
//! run, or a not-yet-encoded image in the recomputed suffix) are excluded by
//! the caller's gate in `prefix_lookup.rs`: the Marconi skip is only taken
//! when every pad lies strictly below the resume point.

use std::borrow::Cow;

/// Virtual cache-key id for the `pos`-th pad token of an image whose content
/// hash is `h`. splitmix64 finalizer over the (hash, position) pair; bit 31
/// forced so the id lies outside every real vocabulary.
#[inline]
fn virtual_pad_id(h: u64, pos: u64) -> u32 {
    let mut z = h ^ pos.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    0x8000_0000u32 | ((z as u32) & 0x7FFF_FFFF)
}

/// Build the cache-key view of `tokens`.
///
/// - No pad tokens present → `Some(Borrowed)` — byte-identical to today, no
///   allocation.
/// - Pads present and `hashes` covers every pad RUN seen (runs are contiguous
///   pad spans; the template renders one run per vision item, in item order)
///   → `Some(Owned)` with every pad substituted.
/// - Pads present but `hashes` is empty or too short (hashes never stamped,
///   EP worker rank, post-swap restore, malformed prompt) → `None`; the
///   caller falls back to the legacy vision veto. Fail-safe: an unusable view
///   disables caching for the request, it never mis-addresses it.
///
/// `tokens` may be any prefix of the full stream (chunk-boundary slices cut
/// a run mid-way): run→hash assignment is by run order from position 0, so a
/// truncated final run substitutes identically to its full-length version.
pub(crate) fn substitute_vision_pads<'a>(
    tokens: &'a [u32],
    image_pad: u32,
    video_pad: u32,
    hashes: &[u64],
) -> Option<Cow<'a, [u32]>> {
    let is_pad = |t: u32| t == image_pad || t == video_pad;
    let first = match tokens.iter().position(|&t| is_pad(t)) {
        None => return Some(Cow::Borrowed(tokens)),
        Some(i) => i,
    };
    if hashes.is_empty() {
        return None;
    }
    let mut out = tokens.to_vec();
    let mut item = 0usize; // run index == vision-item index
    let mut i = first;
    while i < out.len() {
        if is_pad(out[i]) {
            let Some(&h) = hashes.get(item) else {
                // More pad runs than stamped hashes — cannot address soundly.
                return None;
            };
            let run_start = i;
            while i < out.len() && is_pad(out[i]) {
                out[i] = virtual_pad_id(h, (i - run_start) as u64);
                i += 1;
            }
            item += 1;
        } else {
            i += 1;
        }
    }
    Some(Cow::Owned(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG: u32 = 248_056;
    const VID: u32 = 248_057;

    #[test]
    fn text_only_borrows_untouched() {
        let t = vec![1, 2, 3];
        let v = substitute_vision_pads(&t, IMG, VID, &[]).unwrap();
        assert!(matches!(v, Cow::Borrowed(_)));
        assert_eq!(v.as_ref(), &t[..]);
    }

    #[test]
    fn pads_without_hashes_veto() {
        let t = vec![1, IMG, IMG, 2];
        assert!(substitute_vision_pads(&t, IMG, VID, &[]).is_none());
    }

    #[test]
    fn more_runs_than_hashes_veto() {
        let t = vec![IMG, 1, VID, 2];
        assert!(substitute_vision_pads(&t, IMG, VID, &[7]).is_none());
    }

    #[test]
    fn identical_images_produce_identical_views() {
        let t = vec![10, IMG, IMG, IMG, 11, VID, VID, 12];
        let a = substitute_vision_pads(&t, IMG, VID, &[111, 222]).unwrap();
        let b = substitute_vision_pads(&t, IMG, VID, &[111, 222]).unwrap();
        assert_eq!(a.as_ref(), b.as_ref());
        // Non-pad tokens pass through; every pad is out of vocab range.
        assert_eq!(a[0], 10);
        assert_eq!(a[4], 11);
        assert_eq!(a[7], 12);
        assert!(a[1] >= 0x8000_0000 && a[5] >= 0x8000_0000);
    }

    #[test]
    fn different_image_diverges_at_its_first_pad() {
        let t = vec![10, IMG, IMG, 11, IMG, IMG, 12];
        let a = substitute_vision_pads(&t, IMG, VID, &[111, 222]).unwrap();
        let b = substitute_vision_pads(&t, IMG, VID, &[111, 999]).unwrap();
        // Shared prefix (text + first image) identical…
        assert_eq!(a[..4], b[..4]);
        // …second image differs from its very first pad.
        assert_ne!(a[4], b[4]);
    }

    #[test]
    fn truncated_run_is_a_prefix_of_the_full_view() {
        let t = vec![10, IMG, IMG, IMG, IMG, 11];
        let full = substitute_vision_pads(&t, IMG, VID, &[42]).unwrap();
        let cut = substitute_vision_pads(&t[..3], IMG, VID, &[42]).unwrap();
        assert_eq!(&full[..3], cut.as_ref());
    }

    #[test]
    fn positions_within_a_run_differ() {
        let t = vec![IMG, IMG, IMG];
        let v = substitute_vision_pads(&t, IMG, VID, &[7]).unwrap();
        assert_ne!(v[0], v[1]);
        assert_ne!(v[1], v[2]);
    }
}
