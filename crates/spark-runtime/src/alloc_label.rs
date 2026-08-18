// SPDX-License-Identifier: AGPL-3.0-only

//! Scoped allocation labels for the `--mem-report` footprint histogram.
//!
//! # Why a label stack and not a label argument
//!
//! There are ~900 `alloc()` call sites. Threading a label through every one of
//! them would be a mechanical edit of the whole tree that future call sites
//! would silently forget to fill in, and the sites that matter most — the ones
//! nobody thought about — are exactly the ones that would pass `""`. A scoped
//! stack inverts that: a label is declared once at a structural boundary (a
//! loader phase, a layer family) and every allocation underneath it inherits
//! the name, including allocations added later by someone who never heard of
//! this module.
//!
//! # Why there is also a backtrace fallback
//!
//! Inheritance only covers what someone thought to wrap. The whole reason this
//! machinery exists is that a 2.34 GiB bucket and a 1.00 GiB bucket in the
//! footprint histogram were *unattributed* — nobody knew which code allocated
//! them, so nobody could have pre-labelled them. When an allocation large
//! enough to matter arrives with no scope active, we capture a backtrace and
//! name it after the first frame belonging to this workspace. That is slow
//! (~100 µs) and produces uglier names than a hand-written label, which is
//! precisely why it is capped, gated behind the report flag, and applied only
//! above a size threshold — it is the net that catches what the labels missed,
//! not the primary mechanism.
//!
//! # What runs always vs only under the flag
//!
//! The SCOPE STACK is always live: a `&'static str` push/pop and a `Cow`
//! borrow, no allocation. That is deliberate — it makes the load-time
//! `weights ≤ k × checkpoint` tripwire in `factory::build` available on every
//! start rather than only when someone remembers to ask for a report, and a
//! tripwire nobody has to opt into is the only kind that would have caught the
//! three leaks that shipped.
//!
//! The BACKTRACE FALLBACK and the histogram dump run only under
//! `--mem-report`, because those are the parts that cost real time.

use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Master switch. Set by `--mem-report` or `ATLAS_ALLOC_HISTO=1`.
static MEM_REPORT: AtomicBool = AtomicBool::new(false);

/// Backtrace captures spent so far, to bound the cost of a pathological load
/// that allocates thousands of large unlabelled buffers.
static BT_BUDGET: AtomicUsize = AtomicUsize::new(0);

/// Allocations at least this large get a backtrace when no scope is active.
/// Below it the name is not worth ~100 µs: the histogram is read top-down by
/// total bytes, and a bucket under 4 MiB × its count has never been the answer.
const BT_MIN_BYTES: usize = 4 * 1024 * 1024;

/// Hard cap on backtrace captures per process.
const BT_MAX_CAPTURES: usize = 4096;

/// Enable or disable footprint reporting. Call before the GPU backend is
/// constructed — allocations made earlier are not in the ledger at all.
pub fn set_mem_report(on: bool) {
    MEM_REPORT.store(on, Ordering::Relaxed);
}

/// Whether footprint reporting is on.
#[inline]
pub fn mem_report_enabled() -> bool {
    MEM_REPORT.load(Ordering::Relaxed)
}

/// Read the legacy env var. Kept so existing `ATLAS_ALLOC_HISTO=1` invocations
/// (and the runs already recorded against them) keep working after the flag
/// landed.
pub fn mem_report_from_env() -> bool {
    std::env::var("ATLAS_ALLOC_HISTO").as_deref() == Ok("1")
}

thread_local! {
    static STACK: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

/// RAII guard naming every allocation made on this thread while it is alive.
///
/// Nesting is allowed; the innermost label wins, so a coarse phase label can
/// wrap a region that names its own tensors more precisely.
pub struct AllocScope {
    /// Not `()`: the guard must not be constructible outside this module, or a
    /// caller could `drop` a frame it never pushed and unbalance the stack.
    _private: (),
}

impl AllocScope {
    /// Open a scope.
    pub fn new(label: &'static str) -> Self {
        STACK.with(|s| s.borrow_mut().push(label));
        Self { _private: () }
    }
}

impl Drop for AllocScope {
    fn drop(&mut self) {
        STACK.with(|s| {
            s.borrow_mut().pop();
        });
    }
}

/// Open a labelled allocation scope. Hold the returned guard for the region.
///
/// ```ignore
/// let _s = alloc_scope("ssm.out_proj.nvfp4");
/// let w = gpu.alloc(bytes)?;   // shows up under that name in the report
/// ```
#[inline]
pub fn alloc_scope(label: &'static str) -> AllocScope {
    AllocScope::new(label)
}

/// The innermost active label on this thread, if any.
pub fn current_label() -> Option<&'static str> {
    STACK.with(|s| s.borrow().last().copied())
}

/// Name for an allocation of `bytes`.
///
/// Under `--mem-report`, an allocation big enough to matter is named
/// `"{scope} / {call site}"` — the scope alone is too coarse to act on. A first
/// run of this report put 22.89 GiB of a 48.71 GiB footprint into a single
/// `weights.layers` row, which says only "the layers hold the memory": true,
/// useless, and exactly the state the report existed to end. The call site is
/// what turns a row into a file to open.
///
/// The scope stays the PREFIX so that `live_alloc_bytes_under("weights")` keeps
/// matching, and remains the whole label when reporting is off — the tripwire
/// needs the scope on every start, and it must never pay for a backtrace.
pub fn label_for(bytes: usize) -> Cow<'static, str> {
    let scope = current_label();

    if !mem_report_enabled() || bytes < BT_MIN_BYTES {
        return scope.map_or(Cow::Borrowed("unlabelled"), Cow::Borrowed);
    }
    if BT_BUDGET.fetch_add(1, Ordering::Relaxed) >= BT_MAX_CAPTURES {
        return scope.map_or(Cow::Borrowed("unlabelled (backtrace budget spent)"), |s| {
            Cow::Owned(format!("{s} / (backtrace budget spent)"))
        });
    }
    let site = infer_label_from_backtrace();
    Cow::Owned(match scope {
        Some(s) => format!("{s} / {site}"),
        None => site,
    })
}

/// Best-effort caller name from a captured backtrace.
///
/// Parses the `Display` form because stable Rust exposes no frame-by-frame API
/// on `std::backtrace::Backtrace`, and pulling in the `backtrace` crate for a
/// diagnostic that runs behind a flag is not worth a new dependency. Release
/// builds here carry no line tables, so this yields function names only — which
/// is all the histogram needs: it answers "which code allocated this", and the
/// grep to the exact line is then trivial.
fn infer_label_from_backtrace() -> String {
    let bt = std::backtrace::Backtrace::force_capture().to_string();
    for line in bt.lines() {
        let line = line.trim();
        // Frame lines look like `12: some::path::function`; skip the
        // `at /path/file.rs:1` continuation lines.
        let Some((_idx, sym)) = line.split_once(": ") else {
            continue;
        };
        let sym = sym.trim();
        if !is_workspace_frame(sym) {
            continue;
        }
        return tidy_symbol(sym);
    }
    "unlabelled (no workspace frame)".to_string()
}

/// Frames belonging to this workspace, minus the allocator plumbing every
/// allocation passes through (which would otherwise name every bucket
/// `record_alloc_size`).
fn is_workspace_frame(sym: &str) -> bool {
    const OURS: [&str; 6] = [
        "spark_model",
        "spark_runtime",
        "spark_server",
        "atlas_kernels",
        "atlas_core",
        "spark_storage",
    ];
    const PLUMBING: [&str; 6] = [
        "alloc_label",
        "record_alloc_size",
        "record_alloc",
        "AtlasCudaBackend::alloc",
        "gpu_impl",
        "as GpuBackend>::alloc",
    ];
    OURS.iter().any(|o| sym.contains(o)) && !PLUMBING.iter().any(|p| sym.contains(p))
}

/// Strip the trailing codegen hash and the leading crate path noise, keeping
/// the tail that identifies the function.
fn tidy_symbol(sym: &str) -> String {
    let sym = sym.split("::{{closure}}").next().unwrap_or(sym);
    // Drop a trailing `::h1a2b3c4d5e6f7890` codegen hash.
    let parts: Vec<&str> = sym
        .split("::")
        .filter(|p| {
            !(p.len() == 17 && p.starts_with('h') && p[1..].chars().all(|c| c.is_ascii_hexdigit()))
        })
        .collect();
    // Keep the last three path components: enough to disambiguate
    // `qwen35_dense::load_nvfp4` from `qwen35_moe::load_nvfp4` without
    // printing the whole mangled path.
    let n = parts.len();
    parts[n.saturating_sub(3)..].join("::")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `MEM_REPORT` is process-global and cargo runs tests on parallel threads,
    /// so every test that touches the switch takes this lock first. Without it
    /// they race and fail intermittently — the worst kind of test.
    static SWITCH: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Scope tracking is NOT gated on the report flag — the load-time ratio
    /// tripwire depends on labels being present on an ordinary start.
    #[test]
    fn scopes_work_with_the_report_off() {
        let _lock = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
        set_mem_report(false);
        {
            let _outer = alloc_scope("weights");
            assert_eq!(current_label(), Some("weights"));
            // No report ⇒ the scope IS the whole label, and no backtrace is
            // taken even for a gigabyte.
            assert_eq!(label_for(1 << 30), Cow::Borrowed("weights"));
            {
                let _inner = alloc_scope("weights.ssm.out_proj");
                assert_eq!(current_label(), Some("weights.ssm.out_proj"));
            }
            assert_eq!(current_label(), Some("weights"));
        }
        assert_eq!(current_label(), None);
    }

    /// The expensive half is gated on the flag AND on the size.
    #[test]
    fn backtrace_is_gated_on_both_the_flag_and_the_size() {
        let _lock = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
        set_mem_report(false);
        assert_eq!(label_for(1 << 30), Cow::Borrowed("unlabelled"));

        set_mem_report(true);
        assert_eq!(label_for(1024), Cow::Borrowed("unlabelled"));
        set_mem_report(false);
    }

    /// Under the report, a large allocation is named `scope / call site`. The
    /// scope must stay the PREFIX: `live_alloc_bytes_under("weights")` is how
    /// the tripwire measures, and a label that led with the call site would
    /// drop the allocation out of the ratio entirely.
    #[test]
    fn large_scoped_allocs_get_scope_slash_site() {
        let _lock = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
        set_mem_report(true);
        let label = {
            let _s = alloc_scope("weights.layers");
            label_for(64 * 1024 * 1024)
        };
        set_mem_report(false);
        assert!(
            label.starts_with("weights.layers"),
            "scope must remain the prefix, got {label:?}"
        );
        assert!(
            label.contains('/'),
            "expected a call site to be appended, got {label:?}"
        );
    }

    #[test]
    fn tidy_symbol_drops_hash_and_keeps_tail() {
        assert_eq!(
            tidy_symbol("spark_model::weight_loader::qwen35_dense::load_nvfp4::h0123456789abcdef"),
            "weight_loader::qwen35_dense::load_nvfp4"
        );
    }

    #[test]
    fn plumbing_frames_are_not_chosen_as_labels() {
        assert!(!is_workspace_frame(
            "spark_runtime::cuda_backend::AtlasCudaBackend::record_alloc_size"
        ));
        assert!(!is_workspace_frame("core::ptr::drop_in_place"));
        assert!(is_workspace_frame(
            "spark_model::weight_loader::qwen35_dense::load_nvfp4"
        ));
    }
}

/// `--low-memory` / `ATLAS_LOW_MEMORY=1`: trade prefill bandwidth for resident
/// footprint.
///
/// Atlas keeps most projections resident in TWO layouts — the packed `[N,K]`
/// form decode reads, plus a transpose or MMQ repack whose only purpose is
/// coalesced N-dim reads during prefill (`kernels/gb10/common/w4a16_gemm.cu`
/// documents the pair as identical math). On `unsloth/Qwen3.8-27B-NVFP4` the
/// FFN's second layout alone is 8.96 GiB on top of a 21.81 GiB checkpoint,
/// which is the difference between fitting a 24 GB card and not.
///
/// When set, the second layout is never built and prefill dispatches the
/// non-transposed kernel over the same bytes decode uses.
pub fn low_memory() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_LOW_MEMORY").as_deref() == Ok("1"))
}
