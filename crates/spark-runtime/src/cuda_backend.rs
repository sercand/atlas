// SPDX-License-Identifier: AGPL-3.0-only

//! Real CUDA GPU backend using AtlasRegistry.
//!
//! SBIO IORouter: all CUDA operations flow through `GpuBackend`.
//! Uses `AtlasRegistry` for kernel loading/launching and raw CUDA
//! driver API for memory management.

use std::ffi::c_void;

use anyhow::{Result, bail};
use std::sync::Arc;

use atlas_core::registry::AtlasRegistry;

mod fault_probe;
mod gpu_copy;
mod gpu_impl;
mod gpu_impl_graph;

// ── Raw CUDA driver API for memory operations ──

unsafe extern "C" {
    pub(super) fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    pub(super) fn cuMemFree_v2(dptr: u64) -> i32;
    pub(super) fn cuMemcpyHtoDAsync_v2(
        dst: u64,
        src: *const c_void,
        bytes: usize,
        stream: u64,
    ) -> i32;
    pub(super) fn cuMemcpyDtoHAsync_v2(
        dst: *mut c_void,
        src: u64,
        bytes: usize,
        stream: u64,
    ) -> i32;
    pub(super) fn cuMemcpyDtoDAsync_v2(dst: u64, src: u64, bytes: usize, stream: u64) -> i32;
    pub(super) fn cuStreamSynchronize(stream: u64) -> i32;
    pub(super) fn cuStreamQuery(stream: u64) -> i32;
    pub(super) fn cuMemHostGetDevicePointer_v2(
        dptr: *mut u64,
        host: *mut std::ffi::c_void,
        flags: u32,
    ) -> i32;
    pub(super) fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> i32;
    /// Device of the calling context, then any `CUdevice_attribute` on it.
    /// Used for `sm_count` (attribute 16 = MULTIPROCESSOR_COUNT).
    pub(super) fn cuCtxGetDevice(device: *mut i32) -> i32;
    pub(super) fn cuDeviceGetAttribute(pi: *mut i32, attrib: u32, dev: i32) -> i32;
    pub(super) fn cuMemsetD8Async(dst: u64, value: u8, n: usize, stream: u64) -> i32;
    // CUDA graph capture/replay
    pub(super) fn cuStreamBeginCapture(hStream: u64, mode: u32) -> i32;
    // Capture-status query (telemetry taps must not sync/copy inside an
    // active capture). Not declared under SCALE — its libcuda export set is
    // minimal and an unresolved extern would break the gfx1151 link.
    #[cfg(not(atlas_scale))]
    pub(super) fn cuStreamIsCapturing(hStream: u64, captureStatus: *mut u32) -> i32;
    pub(super) fn cuStreamEndCapture(hStream: u64, phGraph: *mut u64) -> i32;
    // CUDA-graph instantiate. NVIDIA's libcuda exports the 3-arg
    // `cuGraphInstantiateWithFlags`; SCALE's libcuda (gfx1151) exports only
    // `cuGraphInstantiate` — same ABI `(CUgraphExec*, CUgraph, u64)`, no
    // `WithFlags` alias. `atlas_scale` (set by build.rs from ATLAS_TARGET_HW)
    // picks the symbol that exists so the binary links on both targets.
    #[cfg(not(atlas_scale))]
    pub(super) fn cuGraphInstantiateWithFlags(
        phGraphExec: *mut u64,
        hGraph: u64,
        flags: u64,
    ) -> i32;
    #[cfg(atlas_scale)]
    pub(super) fn cuGraphInstantiate(phGraphExec: *mut u64, hGraph: u64, flags: u64) -> i32;
    pub(super) fn cuGraphLaunch(hGraphExec: u64, hStream: u64) -> i32;
    pub(super) fn cuGraphExecDestroy(hGraphExec: u64) -> i32;
    pub(super) fn cuGraphDestroy(hGraph: u64) -> i32;
    fn cuCtxGetCurrent(pctx: *mut u64) -> i32;
    pub(super) fn cuCtxSetCurrent(ctx: u64) -> i32;
    pub(super) fn cuStreamCreate(phStream: *mut u64, flags: u32) -> i32;
    // Page-locked host memory for efficient async transfers
    pub(super) fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> i32;
    pub(super) fn cuMemFreeHost(p: *mut c_void) -> i32;
    // Managed (unified) memory — allows over-subscription with Linux swap paging
    pub(super) fn cuMemAllocManaged(dptr: *mut u64, bytesize: usize, flags: u32) -> i32;
    // CUDA events for inter-stream synchronization
    pub(super) fn cuEventCreate(phEvent: *mut u64, flags: u32) -> i32;
    pub(super) fn cuEventRecord(hEvent: u64, hStream: u64) -> i32;
    pub(super) fn cuStreamWaitEvent(hStream: u64, hEvent: u64, flags: u32) -> i32;
    pub(super) fn cuEventSynchronize(hEvent: u64) -> i32;
    pub(super) fn cuEventDestroy_v2(hEvent: u64) -> i32;
}

/// Production GPU backend wrapping AtlasRegistry + raw CUDA driver API.
///
/// **Owns this model's kernel modules.** The registry used to be a process
/// singleton reached through `AtlasRegistry::get()`; it is now loaded per model
/// and propagated from here, so a swapped-in model cannot run the previous
/// model's kernels. Dropping the last backend unloads them.
/// One row of the live-allocation ledger.
pub struct AllocInfo {
    /// Bytes as REQUESTED, not as the driver rounded them. The histogram
    /// buckets on this, and a bucket is only readable as "N copies of one
    /// tensor" if the number is the tensor's own size.
    pub bytes: usize,
    /// Innermost `alloc_scope` active at request time. `Cow` so a static scope
    /// label costs nothing to carry and only a backtrace-inferred name (which
    /// `--mem-report` alone produces) allocates.
    pub label: std::borrow::Cow<'static, str>,
}

pub struct AtlasCudaBackend {
    /// This model's kernel modules. `Arc` because the backend is cloned into
    /// the layers that launch kernels.
    registry: Arc<AtlasRegistry>,
    /// `ATLAS_DEBUG_SYNC_KERNELS=1` — sync after every launch. Read once here
    /// rather than per launch, and carried rather than cached in a static.
    debug_sync_kernels: bool,
    /// This model's kernel handles and op scratch. Dropped with the backend,
    /// so neither can outlive the registry or context it came from.
    op_cache: crate::op_cache::OpCache,
    /// Every device allocation this backend made and has not freed.
    ///
    /// The backend is created per model (`preflight.rs`) and moved into it, so
    /// this ledger is exactly model-scoped: what is still outstanding when the
    /// model is torn down is what that model leaked.
    ///
    /// This exists because enumerating owners does not scale. The loaders
    /// FUSE weights — `qwen35_dense.rs:98` allocates a new buffer and copies
    /// two source tensors into it — and hand the result to a layer struct. The
    /// sources live in `WeightStore` and are released with it; the fused copy
    /// is owned by a `Box<dyn TransformerLayer>` and was released by nothing.
    /// Measured on a 27B: 15.3 GB leaked per load/teardown cycle, linear
    /// across six cycles with no plateau.
    ///
    /// Process-lifetime workspaces are NOT in here and must not be: CUTLASS
    /// (`cutlass.rs:246`) and FlashInfer (`flashinfer.rs:145`) call
    /// `cuMemAlloc_v2` directly rather than through this allocator, so freeing
    /// the ledger cannot invalidate a static that outlives the model.
    /// The ledger carries each allocation's SIZE and OWNER, not just its
    /// pointer, and does so unconditionally. That is what makes "pre-KV is
    /// 2.4x the checkpoint" a number the process can check against itself at
    /// load time instead of a number a human has to go measure — three leaks
    /// shipped because nothing watched that ratio. Cost is one `Cow` beside a
    /// hash entry that already existed; the previous design hashed the pointer
    /// twice (set + histogram map) whenever reporting was on, so this is
    /// strictly cheaper there and equal when off.
    live_allocs: parking_lot::Mutex<std::collections::HashMap<u64, AllocInfo>>,
    /// Default CUDA stream handle (from the process CUDA host).
    default_stream: u64,
    /// CUDA context handle for cross-thread binding.
    cuda_ctx: u64,
}

impl AtlasCudaBackend {
    /// Initialize the CUDA backend on the given GPU ordinal.
    ///
    /// Loads the provided PTX modules for THIS model. Use
    /// `atlas_kernels::ptx_for_model()` or `ptx_modules()` to obtain the
    /// correct module set. Each call produces an independent module set — the
    /// CUDA context and stream are shared, nothing else is.
    pub fn new(ordinal: usize, ptx_modules: &[(&'static str, &'static [u8])]) -> Result<Self> {
        // A new model's GPU state begins here, so the run mailboxes start
        // clean — upstream of the first kernel lookup, so the kernel audit
        // records only this model's modules. See `crate::run_metrics`.
        crate::run_metrics::reset_for_new_run();
        // `--mem-report` sets the switch before we get here; the legacy env var
        // is honoured too, so runs recorded against `ATLAS_ALLOC_HISTO=1` stay
        // reproducible. Never turn it OFF here: that would let backend
        // construction silently undo the flag.
        if crate::alloc_label::mem_report_from_env() {
            crate::alloc_label::set_mem_report(true);
        }
        let registry = AtlasRegistry::load(ordinal, ptx_modules)
            .map_err(|e| anyhow::anyhow!("AtlasRegistry load failed: {e}"))?;
        let default_stream = registry.raw_stream();

        // Capture current CUDA context for cross-thread binding.
        let mut cuda_ctx: u64 = 0;
        let status = unsafe { cuCtxGetCurrent(&mut cuda_ctx) };
        if status != 0 || cuda_ctx == 0 {
            bail!("cuCtxGetCurrent failed: status {status}, ctx {cuda_ctx:#x}");
        }

        tracing::info!(
            "AtlasCudaBackend initialized on GPU {ordinal} with {} PTX modules",
            ptx_modules.len()
        );

        Ok(Self {
            live_allocs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            registry,
            debug_sync_kernels: std::env::var("ATLAS_DEBUG_SYNC_KERNELS").as_deref() == Ok("1"),
            op_cache: crate::op_cache::OpCache::new(),
            default_stream,
            cuda_ctx,
        })
    }

    /// Enter `ptr` in the live ledger with its size and owning scope.
    ///
    /// Resolves the label BEFORE taking the lock: under `--mem-report` an
    /// unlabelled large allocation costs a ~100 µs backtrace, and loaders
    /// allocate from several threads.
    pub(crate) fn record_alloc(&self, ptr: crate::gpu::DevicePtr, bytes: usize) {
        let label = crate::alloc_label::label_for(bytes);
        self.live_allocs
            .lock()
            .insert(ptr.0, AllocInfo { bytes, label });
    }

    pub(crate) fn forget_alloc(&self, ptr: crate::gpu::DevicePtr) {
        self.live_allocs.lock().remove(&ptr.0);
    }

    /// Total bytes this backend has allocated and not freed.
    ///
    /// Always available — the ledger is unconditional. This is Atlas's OWN
    /// footprint by construction, which is why it can be compared to the
    /// checkpoint size without the co-tenant arithmetic that
    /// `free_memory()`-based measures need.
    pub fn live_alloc_bytes(&self) -> Option<usize> {
        Some(self.live_allocs.lock().values().map(|a| a.bytes).sum())
    }

    /// Bytes held by allocations whose owning scope starts with `prefix`.
    ///
    /// The scale-free half of the load-time tripwire: weight-derived bytes are
    /// a function of the checkpoint alone, whereas total pre-KV also contains
    /// arenas and state pools that scale with batch size and would make a
    /// whole-footprint ratio fire on legitimately large configs.
    pub fn live_alloc_bytes_under(&self, prefix: &str) -> usize {
        self.live_allocs
            .lock()
            .values()
            .filter(|a| a.label.starts_with(prefix))
            .map(|a| a.bytes)
            .sum()
    }

    /// DIAGNOSTIC (`--mem-report`): report the LIVE allocation set two ways.
    ///
    /// By owner first — that is the actionable view, because a label names code
    /// you can go delete. Then by exact request size, because a size bucket
    /// names one tensor shape and its `count` reads as "how many copies of this
    /// tensor are resident", which is the question a footprint that exceeds the
    /// checkpoint by 2-3x actually poses. Neither view subsumes the other: the
    /// owner view finds the leaking function, the size view proves how many
    /// duplicates it made.
    pub fn dump_alloc_histo(&self, tag: &str) {
        if !crate::alloc_label::mem_report_enabled() {
            return;
        }
        let ledger = self.live_allocs.lock();
        let total: usize = ledger.values().map(|a| a.bytes).sum();
        let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);

        tracing::warn!(
            "ALLOC HISTO [{tag}]: {} live allocations, {:.2} GiB total",
            ledger.len(),
            gib(total),
        );

        // ── by owner ──
        let mut by_label: std::collections::HashMap<&str, (usize, usize)> =
            std::collections::HashMap::new();
        for info in ledger.values() {
            let e = by_label.entry(info.label.as_ref()).or_insert((0, 0));
            e.0 += info.bytes;
            e.1 += 1;
        }
        let mut label_rows: Vec<(&str, usize, usize)> =
            by_label.into_iter().map(|(l, (b, c))| (l, b, c)).collect();
        label_rows.sort_by_key(|(_, b, _)| std::cmp::Reverse(*b));
        tracing::warn!("  -- by owner --");
        for (label, bytes, count) in label_rows.iter().take(30) {
            tracing::warn!("  {:>8.2} GiB = {:>4} allocs  {}", gib(*bytes), count, label);
        }

        // ── by exact request size ──
        let mut by_size: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for info in ledger.values() {
            *by_size.entry(info.bytes).or_insert(0) += 1;
        }
        let mut rows: Vec<(usize, usize)> = by_size.into_iter().collect();
        rows.sort_by_key(|(bytes, count)| std::cmp::Reverse(bytes * count));
        tracing::warn!("  -- by tensor size --");
        for (bytes, count) in rows.iter().take(30) {
            tracing::warn!(
                "  {:>8.2} GiB = {:>4} x {:>10.2} MiB",
                gib(bytes * count),
                count,
                *bytes as f64 / (1024.0 * 1024.0),
            );
        }
    }

    /// Free every allocation this backend made and nobody released.
    ///
    /// The backstop for allocations no `ModelResource` covers — chiefly the
    /// loaders' fused weights, which are owned by layer structs rather than by
    /// any pool. Returns how many were reclaimed; the ledger now carries each
    /// one's requested size, so the bytes are logged alongside the count — a
    /// count alone cannot distinguish 160 stranded 30 MiB weights from 160
    /// stranded scalars, and that distinction was the whole question.
    ///
    /// Runs LAST in teardown, after every `ModelResource::release`, so it only
    /// ever sees what those missed — and each `free` here has already been
    /// removed from the ledger by `forget_alloc`, so it cannot double-free.
    pub fn sweep_unreleased(&self) -> usize {
        self.dump_alloc_histo("live set at sweep");
        let outstanding: Vec<(u64, AllocInfo)> = self.live_allocs.lock().drain().collect();
        let count = outstanding.len();
        let bytes: usize = outstanding.iter().map(|(_, a)| a.bytes).sum();
        if count > 0 {
            tracing::warn!(
                "sweep: {count} unreleased allocations, {:.2} GiB",
                bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        for (raw, _) in outstanding {
            // Bypass `free`: the ledger is already drained, and a failure here
            // must not abort the rest of the sweep.
            let status = unsafe { cuMemFree_v2(raw) };
            if status != 0 && !atlas_core::registry::is_teardown_noop(status) {
                tracing::warn!("sweep: cuMemFree failed for {raw:#x}: status {status}");
            }
        }
        count
    }

    pub fn registry(&self) -> &Arc<AtlasRegistry> {
        &self.registry
    }

    pub(crate) fn debug_sync_kernels(&self) -> bool {
        self.debug_sync_kernels
    }
}

/// Last-resort reclamation for a backend that never reached model teardown.
///
/// A load that FAILS part-way leaves whatever it had already allocated on the
/// ledger, and no `Model` is ever built to tear down. On a hot-swap that memory
/// is not merely leaked, it is actively harmful: the outgoing model is already
/// gone, and the restore then loads into a budget the dead attempt is still
/// holding. That is not hypothetical — a 35B swap failed at kernel selection
/// and the 27B restore died with "only 14.08 GB remains but 17.38 GB is
/// needed", leaving the server with no model at all.
///
/// On the normal path this frees nothing: `Model::teardown` drains the ledger
/// first, so the sweep finds an empty set. Freeing here is the safe case
/// described in `atlas_core::scope` — nothing is allocating against a backend
/// that is being dropped.
impl Drop for AtlasCudaBackend {
    fn drop(&mut self) {
        let swept = self.sweep_unreleased();
        if swept > 0 {
            // Not necessarily a failure: a load abandoned part-way never
            // reaches `Model::teardown`, and this is where its allocations come
            // back. But on a model that DID serve, teardown has already drained
            // the ledger, so anything here belongs to an owner that never
            // registered — say which without asserting a cause the log cannot
            // know. (An earlier wording claimed "from a load that never
            // completed"; it fired on two perfectly healthy swaps and would
            // have sent an operator hunting a failure that had not happened.)
            tracing::warn!(
                "backend drop reclaimed {swept} allocation(s) that no owner released — \
                 expected if a load was abandoned part-way, otherwise an unregistered owner"
            );
        }
    }
}

// ── OOM Watchdog ────────────────────────────────────────────────────
//
// Background task that polls GPU free memory every `interval` and calls
// `std::process::exit(1)` if it drops below `threshold_bytes`.
// On GB10 unified memory, GPU OOM = system OOM = kernel freeze, so
// killing the process early prevents unrecoverable system hangs.

/// Query GPU free memory without requiring a GpuBackend reference.
/// Safe to call from any thread that shares the CUDA context.
///
/// On unified memory systems (GB10), `cuMemGetInfo` reports Linux's "free" memory
/// which excludes reclaimable buff/cache. This under-reports available memory by
/// 30-50%. We take the max of CUDA's report and `/proc/meminfo` MemAvailable
/// to get the true available memory.
pub fn cuda_free_memory_bytes() -> Option<usize> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    let status = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
    if status != 0 {
        return None;
    }

    // On unified memory, also check MemAvailable from /proc/meminfo.
    // This includes reclaimable buff/cache that CUDA doesn't account for.
    if let Some(mem_available) = system_available_memory_bytes() {
        free = free.max(mem_available);
    }
    Some(free)
}

/// Read MemAvailable from /proc/meminfo (Linux only).
/// Returns None on non-Linux or if parsing fails.
fn system_available_memory_bytes() -> Option<usize> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in contents.lines() {
        if line.starts_with("MemAvailable:") {
            let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// Start a background OOM watchdog that polls GPU memory every `interval`.
/// If free memory drops below `threshold_mb` MB, the process exits immediately.
///
/// Returns a `tokio::task::JoinHandle` — drop it to stop the watchdog (on shutdown).
/// Whether the watchdog is already running.
///
/// STATIC, DELIBERATELY — process lifecycle. The watchdog polls DEVICE free
/// memory, which is a property of the process and its GPU, not of any model:
/// one is correct for the whole process no matter how many models come and go.
/// It is nonetheless spawned from inside the model-dependent startup range
/// (after GPU init, which it needs for a context), so a second load would
/// otherwise start a second watchdog polling the same number and logging the
/// same warning twice. Guarding at the source rather than at the call site
/// means a future swap path cannot get this wrong by forgetting.
static WATCHDOG_RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Start the OOM watchdog, or return `None` if one is already running.
pub fn spawn_oom_watchdog(
    threshold_mb: usize,
    interval: std::time::Duration,
) -> Option<tokio::task::JoinHandle<()>> {
    if WATCHDOG_RUNNING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return None;
    }
    let threshold_bytes = threshold_mb * 1024 * 1024;
    Some(tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        // Track consecutive low-memory readings to avoid false positives
        // during transient allocation spikes.
        let mut consecutive_low = 0u32;
        loop {
            tick.tick().await;
            if let Some(free) = cuda_free_memory_bytes() {
                if free < threshold_bytes {
                    consecutive_low += 1;
                    let free_mb = free / (1024 * 1024);
                    tracing::error!(
                        "OOM watchdog: GPU free memory critically low: {} MB (threshold: {} MB) [{}/3]",
                        free_mb,
                        threshold_mb,
                        consecutive_low,
                    );
                    if consecutive_low >= 3 {
                        tracing::error!(
                            "OOM watchdog: 3 consecutive readings below threshold. \
                             Terminating to prevent system freeze."
                        );
                        // Flush logs before exit
                        std::process::exit(1);
                    }
                } else {
                    consecutive_low = 0;
                }
            }
        }
    }))
}

#[cfg(test)]
mod tests;
