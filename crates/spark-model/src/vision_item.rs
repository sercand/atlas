// SPDX-License-Identifier: AGPL-3.0-only

//! The unit of vision input handed to the model.

/// One vision item: a still image, or a video, ready for the encoder.
///
/// `groups` holds the TEMPORAL GROUPS. A still image has exactly one; a video
/// has `frames / temporal_patch_size`, each group a full `grid_h x grid_w`
/// patch plane built from `temporal_patch_size` consecutive frames. Every
/// group is shaped identically — which is why the ViT consumes them on the
/// same path and needs no notion of time.
///
/// Grouping lives in the TYPE rather than in a parallel `groups_per_item`
/// vector carried alongside the pixels. Two vectors that must agree is
/// precisely the failure this feature is exposed to: the pad run, the encoder
/// rows and the MRoPE position stream all have to describe the same item, and
/// a desync between them is silent — fluent output, wrong answer.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionItem {
    /// Per temporal group: `[grid_h * grid_w, C * temporal_patch_size * patch^2]`.
    pub groups: Vec<Vec<f32>>,
    /// Pre-merge patch grid, identical across this item's groups.
    pub grid_h: usize,
    pub grid_w: usize,
}

impl VisionItem {
    /// A still image: one temporal group.
    pub fn image(pixels: Vec<f32>, grid_h: usize, grid_w: usize) -> Self {
        Self {
            groups: vec![pixels],
            grid_h,
            grid_w,
        }
    }

    /// Temporal extent, in groups. 1 for a still.
    pub fn t_len(&self) -> usize {
        self.groups.len().max(1)
    }

    /// Merged tokens this item occupies in the prompt — the length of its pad
    /// run, and the number of embedding rows the encoder will return for it.
    /// The two are the same number by construction, which is the invariant
    /// the whole splice depends on.
    pub fn pad_count(&self, spatial_merge_size: usize) -> usize {
        let sms = spatial_merge_size.max(1);
        self.t_len() * (self.grid_h / sms) * (self.grid_w / sms)
    }

    /// Deterministic 64-bit content hash (FNV-1a over grid dims + every
    /// group's raw f32 bits). Keys image-hash-aware prefix caching
    /// (`model::vision_cache`): identical pixels+grid → identical hash →
    /// the pad run's virtual cache ids match across requests; any pixel
    /// difference changes the hash and the cached prefix diverges at the
    /// image. FNV is deterministic across processes, so a restart neither
    /// helps nor hurts (the in-memory cache is empty anyway).
    pub fn content_hash(&self) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x1000_0000_01b3;
        let mut h = FNV_OFFSET;
        let mut eat_u64 = |v: u64| {
            for b in v.to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(FNV_PRIME);
            }
        };
        eat_u64(self.grid_h as u64);
        eat_u64(self.grid_w as u64);
        eat_u64(self.groups.len() as u64);
        for g in &self.groups {
            eat_u64(g.len() as u64);
            // Word-at-a-time over the f32 bit patterns (byte-at-a-time FNV
            // over multi-MB pixel buffers is needlessly slow).
            let mut acc = FNV_OFFSET;
            for chunk in g.chunks(2) {
                let mut w = 0u64;
                for (i, f) in chunk.iter().enumerate() {
                    w |= (f.to_bits() as u64) << (32 * i);
                }
                acc = (acc ^ w).wrapping_mul(FNV_PRIME);
            }
            eat_u64(acc);
        }
        h
    }
}
