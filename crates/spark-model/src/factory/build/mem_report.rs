// SPDX-License-Identifier: AGPL-3.0-only

//! Load-time footprint report and the weights-vs-checkpoint tripwire.
//!
//! # Why this exists
//!
//! Three separate load-time leaks — 8.28 GiB and 160 allocations between them —
//! shipped and survived, on a model whose weights are 21 GiB and whose pre-KV
//! footprint was 51 GiB. Every one of them was individually invisible: no test
//! asserted a footprint, the server started fine, and the only symptom was a KV
//! cache smaller than it should have been. Nothing in the process ever compared
//! what was resident against what the checkpoint actually contains.
//!
//! # Why the ratio is taken on WEIGHTS, not on pre-KV
//!
//! Pre-KV mixes two populations with completely different scaling laws:
//!
//!   * weights and their derived copies — a function of the checkpoint alone
//!   * arenas, SSM state pools, ViT scratch — a function of `--max-batch-size`,
//!     `--max-seq-len`, `--ssm-cache-slots`, `--vision-max-pixels`
//!
//! A threshold on their sum is unusable: tight enough to catch an 8 GiB weight
//! leak is also tight enough to fire on a legitimate large-batch config, and
//! loose enough to never false-fire would not have caught any of the three
//! leaks. Restricting the ratio to the `weights.*` scopes gives a quantity that
//! depends on the checkpoint and nothing else, so one constant fits every
//! serve profile.
//!
//! # Why it warns rather than aborts by default
//!
//! The ratio is a heuristic over quantisation schemes that legitimately vary:
//! an FP8 checkpoint requantised to NVFP4 carries a derived copy by design, and
//! a future format could carry two. Refusing to start a server on a heuristic
//! turns a diagnostic into an outage. `ATLAS_MEM_RATIO_STRICT=1` makes it fatal
//! for CI, where an outage is the point.

use anyhow::Result;

/// Default ceiling on `weights-resident / checkpoint-bytes`.
///
/// Calibrated against `unsloth/Qwen3.8-27B-NVFP4`, the worst case in tree: a
/// mixed FP8/NVFP4 checkpoint where the FP8 half is dequantised and requantised
/// at load, so a derived copy of those tensors is expected and legitimate. The
/// three known leaks put that model at 2.4x; healthy it sits near 1.6x. 2.0
/// sits above every legitimate configuration measured and below every leaking
/// one.
const DEFAULT_MAX_RATIO: f64 = 2.0;

/// Outcome of the tripwire, split from the logging so it can be tested without
/// a GPU, a checkpoint, or a tracing subscriber.
#[derive(Debug, PartialEq)]
pub(super) enum Verdict {
    /// No checkpoint bytes or no labelled weight bytes — nothing to compare.
    /// Not a pass: saying "ok" when the measurement is absent is how a broken
    /// tripwire looks exactly like a healthy one.
    Unmeasurable,
    Ok { ratio: f64 },
    Exceeded { ratio: f64, limit: f64 },
}

/// Compare resident weight bytes against checkpoint bytes.
pub(super) fn judge(weight_bytes: usize, checkpoint_bytes: usize, limit: f64) -> Verdict {
    if checkpoint_bytes == 0 || weight_bytes == 0 {
        return Verdict::Unmeasurable;
    }
    let ratio = weight_bytes as f64 / checkpoint_bytes as f64;
    if ratio > limit {
        Verdict::Exceeded { ratio, limit }
    } else {
        Verdict::Ok { ratio }
    }
}

/// The configured ceiling: `ATLAS_MEM_RATIO_MAX` if set and parseable, else
/// [`DEFAULT_MAX_RATIO`].
fn configured_limit() -> f64 {
    std::env::var("ATLAS_MEM_RATIO_MAX")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|r| *r > 0.0)
        .unwrap_or(DEFAULT_MAX_RATIO)
}

/// Dump the footprint report (under `--mem-report`) and run the tripwire
/// (always).
///
/// Called at the point where the model's own allocations are complete and the
/// KV cache has not yet been sized — i.e. exactly the "pre-KV" number the
/// campaign tracks.
pub(super) fn report_and_check(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    checkpoint_bytes: usize,
) -> Result<()> {
    gpu.dump_alloc_histo("pre-KV");

    let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let weight_bytes = gpu.live_alloc_bytes_under("weights");
    let limit = configured_limit();

    if let Some(total) = gpu.live_alloc_bytes() {
        tracing::info!(
            "pre-KV footprint: {:.2} GiB resident, of which {:.2} GiB is weights \
             (checkpoint on disk: {:.2} GiB)",
            gib(total),
            gib(weight_bytes),
            gib(checkpoint_bytes),
        );
    }

    match judge(weight_bytes, checkpoint_bytes, limit) {
        Verdict::Unmeasurable => {
            tracing::debug!(
                "weight/checkpoint ratio not checked: {} labelled weight bytes, \
                 {} checkpoint bytes",
                weight_bytes,
                checkpoint_bytes,
            );
        }
        Verdict::Ok { ratio } => {
            tracing::info!("weight/checkpoint ratio {ratio:.2}x (limit {limit:.2}x)");
        }
        Verdict::Exceeded { ratio, limit } => {
            let msg = format!(
                "weight residency {:.2} GiB is {ratio:.2}x the {:.2} GiB checkpoint \
                 (limit {limit:.2}x). Every byte over 1.0x is a derived copy — a \
                 dequant, a requant, or a transposed repack — and past this line \
                 they have historically been copies nothing reads. Re-run with \
                 --mem-report to see which scope holds them; raise the bar with \
                 ATLAS_MEM_RATIO_MAX if this checkpoint legitimately needs it.",
                gib(weight_bytes),
                gib(checkpoint_bytes),
            );
            if std::env::var("ATLAS_MEM_RATIO_STRICT").as_deref() == Ok("1") {
                anyhow::bail!("{msg}");
            }
            tracing::warn!("{msg}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Qwen3.8-27B numbers this was calibrated on: 21.1 GiB of checkpoint.
    const CKPT: usize = 22_656_547_226; // 21.1 GiB

    #[test]
    fn healthy_load_passes() {
        // 1.6x — weights plus the one legitimate FP8->NVFP4 derived copy.
        let verdict = judge((CKPT as f64 * 1.6) as usize, CKPT, DEFAULT_MAX_RATIO);
        assert!(matches!(verdict, Verdict::Ok { .. }), "{verdict:?}");
    }

    /// The state the three shipped leaks actually produced. If this ever passes,
    /// the tripwire has been loosened past the point of usefulness.
    #[test]
    fn the_leaks_that_shipped_would_have_tripped_it() {
        let verdict = judge((CKPT as f64 * 2.4) as usize, CKPT, DEFAULT_MAX_RATIO);
        assert!(matches!(verdict, Verdict::Exceeded { .. }), "{verdict:?}");
    }

    /// Absence of a measurement must not read as a pass.
    #[test]
    fn missing_inputs_are_unmeasurable_not_ok() {
        assert_eq!(judge(0, CKPT, 2.0), Verdict::Unmeasurable);
        assert_eq!(judge(CKPT, 0, 2.0), Verdict::Unmeasurable);
    }

    #[test]
    fn limit_is_exclusive_at_the_boundary() {
        assert!(matches!(judge(200, 100, 2.0), Verdict::Ok { .. }));
        assert!(matches!(judge(201, 100, 2.0), Verdict::Exceeded { .. }));
    }
}
