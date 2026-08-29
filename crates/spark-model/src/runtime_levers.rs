// SPDX-License-Identifier: AGPL-3.0-only

//! Process-global RUNTIME experiment levers.
//!
//! Each lever shadows a boot-time env gate so a perf/numerics A/B can flip it
//! over the admin API (`POST /admin/levers`) without a service bounce — on a
//! box where a bounce costs ~2.5 min of load and evicts every cache. A lever
//! holds a tri-state: *unset* (follow the env var it shadows, i.e. exactly
//! the pre-lever behavior), *forced on*, *forced off*.
//!
//! Levers gate DISPATCH between already-loaded kernels or already-supported
//! paths only — flipping one mid-request changes which kernel the next layer
//! call takes, every variant of which is output-complete, so the worst case
//! of a mid-prefill flip is a chunk whose layers straddle two numerics
//! profiles (same class of effect as the documented batch-K T=0 floor).
//! Levers must never gate memory layout, allocation, or state shape.
//!
//! Born 2026-08-29 (prefill-throughput campaign): the MoE FP4-MMA prefill
//! kernels and the QSA dense-skip needed one-variable A/Bs, and each env-var
//! flip would otherwise have been a restart.

use std::sync::atomic::{AtomicU8, Ordering};

/// One tri-state runtime lever shadowing `env`.
pub struct RtLever {
    /// 0 = unset (read `env` each call), 1 = forced off, 2 = forced on.
    state: AtomicU8,
    env: &'static str,
    /// Admin-facing name (kebab-case-free, snake_case).
    name: &'static str,
}

impl RtLever {
    const fn new(name: &'static str, env: &'static str) -> Self {
        Self {
            state: AtomicU8::new(0),
            env,
            name,
        }
    }

    /// Current value: the override if set, else the shadowed env var
    /// (`"1"` = on, anything else = off — matching the gates it replaced).
    /// The env read is NOT cached: unset levers keep honoring an operator's
    /// boot env exactly as before, and the read is nanoseconds against the
    /// per-chunk work behind every call site.
    #[inline]
    pub fn get(&self) -> bool {
        match self.state.load(Ordering::Relaxed) {
            1 => false,
            2 => true,
            _ => std::env::var(self.env).ok().as_deref() == Some("1"),
        }
    }

    /// Set the override (`None` clears back to env-following).
    pub fn set(&self, v: Option<bool>) {
        let s = match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        };
        self.state.store(s, Ordering::Relaxed);
        tracing::info!(
            "runtime lever {}: {} (env {} currently {})",
            self.name,
            match v {
                None => "cleared -> env",
                Some(true) => "forced ON",
                Some(false) => "forced OFF",
            },
            self.env,
            std::env::var(self.env).ok().as_deref().unwrap_or("unset"),
        );
    }

    fn describe(&self) -> serde_json::Value {
        let state = match self.state.load(Ordering::Relaxed) {
            1 => "forced_off",
            2 => "forced_on",
            _ => "env",
        };
        serde_json::json!({
            "name": self.name,
            "env": self.env,
            "state": state,
            "effective": self.get(),
        })
    }
}

/// Prefill routed-expert gate_up on the block-scaled FP4-MMA kernel
/// (`moe_w4a16_fused_gate_up_t_k64_fp4`) instead of the FP8-MMA w4a16 one.
pub static MOE_GATEUP_FP4: RtLever =
    RtLever::new("moe_gateup_fp4", "ATLAS_HOLO_MOE_GATEUP_FP4");

/// Prefill routed-expert down on the FP4-MMA kernel (`moe_w4a16_down_t_k64_fp4`).
pub static MOE_DOWN_FP4: RtLever = RtLever::new("moe_down_fp4", "ATLAS_HOLO_MOE_DOWN_FP4");

/// Prefill routed-expert down via FP8 activations
/// (`bf16_to_fp8` + `moe_fp8_grouped_gemm_ptrtable_t`).
pub static MOE_FP8_DOWN: RtLever = RtLever::new("moe_fp8_down", "ATLAS_MOE_PREFILL_FP8_DOWN");

/// Restore the compute-then-discard dense attention pass on fully-selective
/// QSA chunks (the pre-campaign behavior), for A/B against the skip.
pub static QSA_KEEP_DENSE_PREFILL: RtLever =
    RtLever::new("qsa_keep_dense_prefill", "ATLAS_QSA_KEEP_DENSE_PREFILL");

/// Prefill routed-expert gate_up on the single-launch CUTLASS grouped NVFP4
/// collective. Engages only when the load-time SFB tables were built
/// (`ATLAS_HOLO_MOE_GROUPED_CUTLASS=1` at boot) — otherwise the dispatch
/// falls through regardless of this lever.
pub static MOE_GROUPED_CUTLASS: RtLever =
    RtLever::new("moe_grouped_cutlass", "ATLAS_HOLO_MOE_GROUPED_CUTLASS");

/// Prefill routed-expert down on the CUTLASS grouped collective (requires
/// the gate_up lever's tables plus a built down table).
pub static MOE_GROUPED_DOWN: RtLever =
    RtLever::new("moe_grouped_down", "ATLAS_HOLO_MOE_GROUPED_DOWN");

/// Restore the host-side QSA prefill top-k (sync D2H of the whole score matrix
/// + multi-threaded quickselect) in place of the on-device `qsa_topk_rows`.
/// Both paths fill the SAME `lists` region of the SAME scratch with the SAME
/// block ids in the SAME order — this selects who computes it, nothing else.
pub static QSA_HOST_TOPK: RtLever = RtLever::new("qsa_host_topk", "ATLAS_QSA_HOST_TOPK");

/// Restore the untiled `qsa_score_rows` (one CTA per (row, block), re-reading
/// the row's query tile for every block) in place of `qsa_score_rows_tiled`.
/// Both produce bit-identical scores into the same buffer.
pub static QSA_SCORE_UNTILED: RtLever =
    RtLever::new("qsa_score_untiled", "ATLAS_QSA_SCORE_UNTILED");

/// Disable the exact-length SSM snapshot fallback: when the only anchor sits at
/// EXACTLY the prompt length (an identical prompt replayed), go back to
/// recomputing the whole prompt instead of restoring the deepest anchor STRICTLY
/// below it. On by default (the fallback is the engine's ordinary warm-turn
/// restore path); this exists to A/B its output against a full recompute without
/// a bounce.
pub static SSM_NO_EXACT_FALLBACK: RtLever =
    RtLever::new("ssm_no_exact_fallback", "ATLAS_SSM_NO_EXACT_FALLBACK");

/// All levers, for the admin list/set endpoints.
pub fn all() -> [&'static RtLever; 9] {
    [
        &MOE_GATEUP_FP4,
        &MOE_DOWN_FP4,
        &MOE_FP8_DOWN,
        &QSA_KEEP_DENSE_PREFILL,
        &MOE_GROUPED_CUTLASS,
        &MOE_GROUPED_DOWN,
        &QSA_HOST_TOPK,
        &QSA_SCORE_UNTILED,
        &SSM_NO_EXACT_FALLBACK,
    ]
}

/// JSON description of every lever (admin GET).
pub fn describe_all() -> serde_json::Value {
    serde_json::Value::Array(all().iter().map(|l| l.describe()).collect())
}

/// Set one lever by admin name. `None` clears to env-following.
/// Returns false when no lever has that name.
pub fn set_by_name(name: &str, v: Option<bool>) -> bool {
    for l in all() {
        if l.name == name {
            l.set(v);
            return true;
        }
    }
    false
}
