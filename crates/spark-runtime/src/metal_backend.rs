// SPDX-License-Identifier: AGPL-3.0-only

//! Apple Metal GPU backend.
//!
//! Implements [`GpuBackend`] on top of the Metal framework via the
//! `objc2-metal` bindings. Apple Silicon is unified-memory (UMA), so
//! every `MTLBuffer` is allocated with `StorageModeShared` and host
//! `memcpy` against `buffer.contents()` is the canonical H2D/D2H path
//! — no PCIe staging, no pinned-host bounce.
//!
//! # Pointer model
//!
//! `DevicePtr` carries a real GPU virtual address obtained from
//! `MTLBuffer::gpuAddress()` (Metal 3+, native to all Apple Silicon).
//! That makes pointer arithmetic (`DevicePtr::offset`) a plain integer
//! add — no buffer/offset pair to thread through. To recover the
//! owning `MTLBuffer` for `free` / blit-copy / `setBuffer:`, we keep
//! a side table `BTreeMap<base_gpu_address, MTLBuffer>` and look up
//! the largest key ≤ ptr. The buffer's gpuAddress range is
//! `[base, base + length)`, so a binary search is enough.
//!
//! # Streams
//!
//! A stream handle indexes a slab of `MetalStream { queue, in_flight,
//! encoder }`. Handle 0 is the default stream and is lazily created on
//! first use. `synchronize(stream)` commits the in-flight
//! `MTLCommandBuffer` and `waitUntilCompleted()`s; the next encoder
//! opens on a fresh buffer.
//!
//! Dispatches accumulate on ONE serial `MTLComputeCommandEncoder` per
//! stream (`MetalStream::encoder`), reused across `launch_typed` calls
//! — encoder churn, not kernel time, dominated per-token latency when
//! every launch opened its own encoder. The serial dispatch type
//! guarantees dispatch N+1 sees dispatch N's writes, so this is
//! semantically identical to the old encoder-per-launch shape. The
//! encoder is ended before anything that can't live inside a compute
//! pass: blits, event signal/wait, and commit.
//!
//! Buffers are bound with `setBuffer:offset:atIndex:`, which makes them
//! resident for the pass automatically — kernels take every input as a
//! typed argument (no argument-buffer pointer chasing), so no
//! `useResource:` calls are needed.
//!
//! # Kernel handles
//!
//! `KernelHandle` indexes a slab of `MTLComputePipelineState`. The
//! library cache (one `MTLLibrary` per `metallib_modules()` entry) is
//! built once at construction and never mutated; pipeline lookups go
//! through the slab + a `(module, fn_name)` HashMap so repeated
//! `kernel()` calls are O(1) cached.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLEvent, MTLLibrary, MTLResourceOptions, MTLSharedEvent, MTLSize,
};
use parking_lot::Mutex;

use crate::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};

// ── Internal type aliases (Retained<ProtocolObject<dyn _>> is verbose) ────

type ObjDevice = Retained<ProtocolObject<dyn MTLDevice>>;
type ObjBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type ObjQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type ObjCmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type ObjLibrary = Retained<ProtocolObject<dyn MTLLibrary>>;
type ObjPipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type ObjSharedEvent = Retained<ProtocolObject<dyn MTLSharedEvent>>;
type ObjComputeEnc = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

// ── Stream + slab types ──────────────────────────────────────────────────

struct MetalStream {
    queue: ObjQueue,
    /// In-flight command buffer accumulating encoded work. Committed +
    /// waited on by `synchronize()`; replaced by a fresh buffer on
    /// next encoder open.
    in_flight: Option<ObjCmdBuf>,
    /// Open compute encoder on `in_flight`, reused across dispatches.
    /// Must be ended before blit encoding, event signal/wait, or commit.
    encoder: Option<ObjComputeEnc>,
}

impl MetalStream {
    /// Borrow (or open) the in-flight command buffer.
    fn cmd_buf(&mut self) -> Result<ObjCmdBuf> {
        if let Some(ref cb) = self.in_flight {
            return Ok(cb.clone());
        }
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("commandBuffer returned null"))?;
        self.in_flight = Some(cb.clone());
        Ok(cb)
    }

    /// Borrow (or open) the stream's serial compute encoder.
    fn compute_encoder(&mut self) -> Result<ObjComputeEnc> {
        if let Some(ref enc) = self.encoder {
            return Ok(enc.clone());
        }
        let cb = self.cmd_buf()?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("computeCommandEncoder returned null"))?;
        self.encoder = Some(enc.clone());
        Ok(enc)
    }

    /// End the open compute encoder, if any. Required before blits,
    /// event encoding, and commit.
    fn end_encoder(&mut self) {
        if let Some(enc) = self.encoder.take() {
            enc.endEncoding();
        }
    }

    /// End encoding and commit the in-flight buffer (no wait). Returns
    /// the committed buffer so callers can `waitUntilCompleted()`.
    fn commit(&mut self) -> Option<ObjCmdBuf> {
        self.end_encoder();
        let cb = self.in_flight.take()?;
        cb.commit();
        Some(cb)
    }
}

/// Tracks one outstanding shared event so `record_event` can write the
/// next signal value and `stream_wait_event` can wait on the same
/// counter.
struct EventSlot {
    event: ObjSharedEvent,
    /// Monotonic value sequence — record_event signals `next`, then
    /// increments. stream_wait_event waits on `next - 1` (the most
    /// recently recorded value). Atomic via the surrounding Mutex.
    next: u64,
}

/// Key for the pipeline cache. Stored as owned strings because the
/// `&str` arguments to `kernel()` come from arbitrary call sites.
type PipelineKey = (String, String);

// ── MetalGpuBackend struct + state ───────────────────────────────────────

pub struct MetalGpuBackend {
    device: ObjDevice,
    /// Side table mapping a buffer's base gpuAddress to the owning
    /// `MTLBuffer`. BTreeMap so we can find the buffer containing an
    /// arbitrary `DevicePtr` via `range(..=ptr).next_back()`.
    allocations: Arc<Mutex<BTreeMap<u64, ObjBuffer>>>,
    /// Stream slab. Indexed by `stream_handle - 1`; handle 0 is the
    /// implicit default stream materialized lazily into slot 0.
    streams: Arc<Mutex<Vec<MetalStream>>>,
    /// Loaded metallibs keyed by module name.
    libraries: HashMap<String, ObjLibrary>,
    /// Pipeline-state cache + slab. The HashMap maps `(module, fn)` to
    /// the slab index; the slab owns the `MTLComputePipelineState`.
    /// Both are mutexed so `kernel()` can be called from any thread.
    pipeline_cache: Arc<Mutex<HashMap<PipelineKey, KernelHandle>>>,
    pipeline_slab: Arc<Mutex<Vec<ObjPipeline>>>,
    /// Function name per slab entry (same index) — profile-mode labels.
    pipeline_names: Arc<Mutex<Vec<String>>>,
    /// Shared-event slab for cross-stream synchronization.
    events: Arc<Mutex<Vec<EventSlot>>>,
}

unsafe impl Send for MetalGpuBackend {}
unsafe impl Sync for MetalGpuBackend {}

impl MetalGpuBackend {
    /// Initialize the Metal backend with the embedded metallib modules.
    ///
    /// `kernel_modules` is the `metallib_modules()` slice produced by
    /// `atlas-kernels`' build script — `(module_name, metallib_bytes)`.
    /// Each entry is loaded into its own `MTLLibrary` via
    /// `newLibraryWithData_error:`. The default stream (handle 0) is
    /// materialized eagerly so the first launch doesn't pay queue-
    /// creation latency.
    pub fn new(ordinal: usize, kernel_modules: &[(&'static str, &'static [u8])]) -> Result<Self> {
        if ordinal != 0 {
            bail!(
                "Metal: only ordinal 0 is supported (Apple Silicon has one \
                 system default device); requested ordinal {ordinal}"
            );
        }
        let device: ObjDevice = MTLCreateSystemDefaultDevice().ok_or_else(|| {
            anyhow!("MTLCreateSystemDefaultDevice returned null — no Metal-capable GPU")
        })?;

        // Build the library cache up-front. `newLibraryWithData_error`
        // takes a `DispatchData`, which is libdispatch's reference-
        // counted byte container. We wrap the &'static slice via
        // dispatch2::DispatchData (zero-copy) — the metallibs are
        // embedded by include_bytes! and outlive the backend.
        let mut libraries: HashMap<String, ObjLibrary> = HashMap::new();
        for (name, bytes) in kernel_modules {
            let data = dispatch2::DispatchData::from_static_bytes(bytes);
            let lib = device.newLibraryWithData_error(&data).map_err(|e| {
                anyhow!(
                    "newLibraryWithData failed for module '{name}': {}",
                    e.localizedDescription()
                )
            })?;
            libraries.insert((*name).to_string(), lib);
        }

        // Materialize the default stream eagerly (slot 0 = handle 0).
        let default_queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("newCommandQueue returned null on default device"))?;
        let streams = vec![MetalStream {
            queue: default_queue,
            in_flight: None,
            encoder: None,
        }];

        tracing::info!(
            "MetalGpuBackend initialized on device '{}' with {} metallib modules",
            device.name().to_string(),
            libraries.len()
        );

        Ok(Self {
            device,
            allocations: Arc::new(Mutex::new(BTreeMap::new())),
            streams: Arc::new(Mutex::new(streams)),
            libraries,
            pipeline_cache: Arc::new(Mutex::new(HashMap::new())),
            pipeline_slab: Arc::new(Mutex::new(Vec::new())),
            pipeline_names: Arc::new(Mutex::new(Vec::new())),
            events: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Return the underlying `MTLDevice` (escape hatch for advanced
    /// use cases — graph capture, custom resource creation, etc.).
    pub fn raw_device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    // ── Internal helpers ────────────────────────────────────────────

    /// Look up the `MTLBuffer` owning `ptr` and the byte offset of
    /// `ptr` within it. Returns `None` if no allocation contains it.
    fn find_buffer(
        allocs: &BTreeMap<u64, ObjBuffer>,
        ptr: DevicePtr,
    ) -> Option<(ObjBuffer, usize)> {
        let (base, buf) = allocs.range(..=ptr.0).next_back()?;
        let offset = (ptr.0 - *base) as usize;
        if offset > buf.length() {
            return None;
        }
        Some((buf.clone(), offset))
    }

    /// Resolve a stream handle to its slab index. Handle 0 → slot 0
    /// (default stream); other handles index `handle - 1`.
    fn stream_index(handle: u64, slab: &[MetalStream]) -> Result<usize> {
        let idx = if handle == 0 {
            0
        } else {
            (handle - 1) as usize
        };
        if idx >= slab.len() {
            bail!("Metal: invalid stream handle {handle}");
        }
        Ok(idx)
    }

    /// Run `f` with exclusive access to the stream's state. Encoding
    /// happens under the streams mutex — cheap now that a launch is a
    /// handful of `set*` calls, and it keeps the encoder cache
    /// consistent without per-stream locks.
    fn with_stream<R>(
        &self,
        stream_handle: u64,
        f: impl FnOnce(&mut MetalStream) -> Result<R>,
    ) -> Result<R> {
        let mut slab = self.streams.lock();
        let idx = Self::stream_index(stream_handle, &slab)?;
        f(&mut slab[idx])
    }
}

// ── GpuBackend impl ──────────────────────────────────────────────────────

impl GpuBackend for MetalGpuBackend {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        // StorageModeShared is the UMA-friendly mode: `contents()`
        // returns a CPU-mappable pointer that aliases GPU memory.
        let buf: ObjBuffer = self
            .device
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("newBufferWithLength failed for {bytes} bytes"))?;
        let addr = buf.gpuAddress();
        if addr == 0 {
            bail!("MTLBuffer::gpuAddress returned 0 — Metal 3 / macOS 13 required");
        }
        self.allocations.lock().insert(addr, buf);
        Ok(DevicePtr(addr))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        // Apple Silicon UMA: managed and shared are the same thing.
        // No paged virtual memory swap mechanism (cuMemAllocManaged on
        // GB10) — Metal lets the OS handle pressure via its memory
        // pool. Defer to plain alloc.
        self.alloc(bytes)
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        // Removing the entry drops the last `Retained` reference; the
        // ObjC runtime releases the underlying MTLBuffer.
        self.allocations.lock().remove(&ptr.0);
        Ok(())
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_h2d: ptr {dst} not in any allocation"))?;
        if offset + src.len() > buf.length() {
            bail!(
                "copy_h2d: write overflows buffer ({} + {} > {})",
                offset,
                src.len(),
                buf.length()
            );
        }
        let contents: NonNull<c_void> = buf.contents();
        unsafe {
            let dst_ptr = (contents.as_ptr() as *mut u8).add(offset);
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr, src.len());
        }
        Ok(())
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, src)
            .ok_or_else(|| anyhow!("copy_d2h: ptr {src} not in any allocation"))?;
        if offset + dst.len() > buf.length() {
            bail!(
                "copy_d2h: read overflows buffer ({} + {} > {})",
                offset,
                dst.len(),
                buf.length()
            );
        }
        let contents: NonNull<c_void> = buf.contents();
        unsafe {
            let src_ptr = (contents.as_ptr() as *const u8).add(offset);
            std::ptr::copy_nonoverlapping(src_ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn copy_d2h_on_stream(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        // UMA: synchronize the stream so prior kernels have written
        // their bytes back through the cache, then memcpy.
        self.synchronize(stream)?;
        self.copy_d2h(src, dst)
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (src_buf, src_off) = Self::find_buffer(&allocs, src)
            .ok_or_else(|| anyhow!("copy_d2d: src ptr {src} not allocated"))?;
        let (dst_buf, dst_off) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_d2d: dst ptr {dst} not allocated"))?;
        drop(allocs);

        let committed = self.with_stream(0, |s| {
            s.end_encoder();
            let cmd_buf = s.cmd_buf()?;
            let enc = cmd_buf
                .blitCommandEncoder()
                .ok_or_else(|| anyhow!("blitCommandEncoder returned null"))?;
            unsafe {
                enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &src_buf, src_off, &dst_buf, dst_off, bytes,
                );
            }
            enc.endEncoding();
            // Synchronize so the d2d behaves like CUDA's synchronous variant.
            Ok(s.commit())
        })?;
        if let Some(cb) = committed {
            cb.waitUntilCompleted();
        }
        Ok(())
    }

    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (src_buf, src_off) = Self::find_buffer(&allocs, src)
            .ok_or_else(|| anyhow!("copy_d2d_async: src {src} not allocated"))?;
        let (dst_buf, dst_off) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_d2d_async: dst {dst} not allocated"))?;
        drop(allocs);
        self.with_stream(stream, |s| {
            s.end_encoder();
            let cmd_buf = s.cmd_buf()?;
            let enc = cmd_buf
                .blitCommandEncoder()
                .ok_or_else(|| anyhow!("blitCommandEncoder returned null"))?;
            unsafe {
                enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &src_buf, src_off, &dst_buf, dst_off, bytes,
                );
            }
            enc.endEncoding();
            Ok(())
        })
    }

    fn launch(
        &self,
        _func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        _params: &mut [*mut c_void],
    ) -> Result<()> {
        // Metal can't safely interpret untyped `*mut c_void` slots as
        // either buffers or scalars (CUDA gets away with this because
        // the driver cross-references the kernel signature). Callers
        // must use `launch_typed`; the cuda-style untyped path is
        // intentionally unsupported here.
        bail!(
            "Metal backend: launch() requires typed args. Use launch_typed() \
             with KernelArg::Buffer / KernelArg::Bytes — see KernelLaunch builder."
        );
    }

    fn launch_typed(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        _shared_mem: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        // Resolve the pipeline state.
        let pipeline = {
            let slab = self.pipeline_slab.lock();
            slab.get(func.0 as usize)
                .cloned()
                .ok_or_else(|| anyhow!("launch_typed: unknown kernel handle {}", func.0))?
        };

        // Resolve only the buffers this launch actually binds. Bound
        // buffers are made resident by `setBuffer:`; nothing else needs
        // `useResource:` (kernels take every input as a typed arg).
        const MAX_ARGS: usize = 16;
        if args.len() > MAX_ARGS {
            bail!("launch_typed: {} args exceeds MAX_ARGS {MAX_ARGS}", args.len());
        }
        let mut resolved: [Option<(ObjBuffer, usize)>; MAX_ARGS] = std::array::from_fn(|_| None);
        {
            let allocs = self.allocations.lock();
            for (idx, arg) in args.iter().enumerate() {
                if let KernelArg::Buffer(p) = arg {
                    let (buf, offset) = Self::find_buffer(&allocs, *p)
                        .ok_or_else(|| anyhow!("launch_typed: arg #{idx} ptr {p} not allocated"))?;
                    resolved[idx] = Some((buf, offset));
                }
            }
        }

        self.with_stream(stream, |s| {
            let enc = s.compute_encoder()?;
            enc.setComputePipelineState(&pipeline);
            for (idx, arg) in args.iter().enumerate() {
                match arg {
                    KernelArg::Buffer(_) => {
                        let (buf, offset) =
                            resolved[idx].as_ref().expect("resolved buffer arg");
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(buf), *offset, idx);
                        }
                    }
                    KernelArg::Bytes(b) => {
                        let ptr = NonNull::new(b.as_ptr() as *mut c_void)
                            .ok_or_else(|| anyhow!("launch_typed: arg #{idx} bytes is null"))?;
                        unsafe {
                            enc.setBytes_length_atIndex(ptr, b.len(), idx);
                        }
                    }
                }
            }

            let threadgroups = MTLSize {
                width: grid[0] as usize,
                height: grid[1] as usize,
                depth: grid[2] as usize,
            };
            let threads_per_tg = MTLSize {
                width: block[0] as usize,
                height: block[1] as usize,
                depth: block[2] as usize,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
            Ok(())
        })?;

        // Profile mode: serialize every launch and attribute its GPU
        // time to the kernel name. Slow — measurement only.
        if profile_enabled() {
            let committed = self.with_stream(stream, |s| Ok(s.commit()))?;
            if let Some(cb) = committed {
                cb.waitUntilCompleted();
                let busy = cb.GPUEndTime() - cb.GPUStartTime();
                let name = self
                    .pipeline_names
                    .lock()
                    .get(func.0 as usize)
                    .cloned()
                    .unwrap_or_else(|| format!("handle{}", func.0));
                profile_record(&name, busy);
            }
        }
        Ok(())
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        let committed = self.with_stream(stream, |s| Ok(s.commit()))?;
        if let Some(cb) = committed {
            cb.waitUntilCompleted();
            record_cb_timing(&cb);
        }
        Ok(())
    }

    fn flush(&self, stream: u64) -> Result<()> {
        // Commit without waiting — lets the GPU start on the encoded
        // prefix while the host keeps encoding into a fresh buffer.
        self.with_stream(stream, |s| {
            s.commit();
            Ok(())
        })
    }

    fn default_stream(&self) -> u64 {
        0
    }

    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        let key: PipelineKey = (module.to_string(), func_name.to_string());
        if let Some(handle) = self.pipeline_cache.lock().get(&key) {
            return Ok(*handle);
        }
        let lib = self
            .libraries
            .get(module)
            .ok_or_else(|| anyhow!("Metal: unknown module '{module}'"))?;
        let ns_name = NSString::from_str(func_name);
        let function = lib.newFunctionWithName(&ns_name).ok_or_else(|| {
            anyhow!("Metal: function '{func_name}' not found in module '{module}'")
        })?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                anyhow!(
                    "newComputePipelineStateWithFunction failed for '{func_name}': {}",
                    e.localizedDescription()
                )
            })?;
        let mut slab = self.pipeline_slab.lock();
        let handle = KernelHandle(slab.len() as u64);
        slab.push(pipeline);
        drop(slab);
        self.pipeline_names.lock().push(func_name.to_string());
        self.pipeline_cache.lock().insert(key, handle);
        Ok(handle)
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        // On UMA we can write through `contents()` directly when we
        // own the whole range — much cheaper than a blit fillBuffer.
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, ptr)
            .ok_or_else(|| anyhow!("memset: ptr {ptr} not allocated"))?;
        if offset + bytes > buf.length() {
            bail!(
                "memset: range overflows buffer ({} + {} > {})",
                offset,
                bytes,
                buf.length()
            );
        }
        let contents = buf.contents();
        unsafe {
            let dst = (contents.as_ptr() as *mut u8).add(offset);
            std::ptr::write_bytes(dst, value, bytes);
        }
        Ok(())
    }

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, _stream: u64) -> Result<()> {
        // UMA + StorageModeShared makes the synchronous memset semantically
        // equivalent (no host/device cache split to flush).
        self.memset(ptr, value, bytes)
    }

    fn total_memory(&self) -> Result<usize> {
        // On Apple Silicon UMA, "device memory" = system RAM. Probe
        // hw.memsize via sysctl for the authoritative number; fall
        // back to MTLDevice.recommendedMaxWorkingSetSize otherwise.
        Ok(sysctl_memsize().unwrap_or_else(|| self.device.recommendedMaxWorkingSetSize() as usize))
    }

    fn free_memory(&self) -> Result<usize> {
        // No direct API for "free GPU memory" on UMA. Approximate via
        // `recommendedMaxWorkingSetSize - currentAllocatedSize`,
        // which matches the headroom Metal will let us allocate
        // before performance degrades.
        let max = self.device.recommendedMaxWorkingSetSize() as usize;
        let used = self.device.currentAllocatedSize();
        Ok(max.saturating_sub(used))
    }

    fn create_stream(&self) -> Result<u64> {
        let queue = self
            .device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("newCommandQueue returned null"))?;
        let mut slab = self.streams.lock();
        slab.push(MetalStream {
            queue,
            in_flight: None,
            encoder: None,
        });
        // Handle = slab index + 1 so handle 0 stays reserved for
        // the default stream.
        Ok(slab.len() as u64)
    }

    fn bind_to_thread(&self) -> Result<()> {
        // Metal devices/queues are thread-safe; no binding required.
        Ok(())
    }

    fn create_event(&self) -> Result<u64> {
        let event = self
            .device
            .newSharedEvent()
            .ok_or_else(|| anyhow!("newSharedEvent returned null"))?;
        let mut slab = self.events.lock();
        slab.push(EventSlot { event, next: 1 });
        // Handle = slab index + 1 (0 reserved for "no event").
        Ok(slab.len() as u64)
    }

    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        let value = {
            let mut slab = self.events.lock();
            let idx = (event as usize)
                .checked_sub(1)
                .ok_or_else(|| anyhow!("record_event: invalid event handle {event}"))?;
            let slot = slab
                .get_mut(idx)
                .ok_or_else(|| anyhow!("record_event: event handle {event} out of range"))?;
            let v = slot.next;
            slot.next += 1;
            v
        };
        let event_obj = {
            let slab = self.events.lock();
            slab[(event - 1) as usize].event.clone()
        };
        // Encode the signal on the active command buffer. Metal will
        // signal value=`value` once everything queued on this buffer
        // up to this point has completed. Event encoding must happen
        // outside any active encoder.
        self.with_stream(stream, |s| {
            s.end_encoder();
            let cmd_buf = s.cmd_buf()?;
            let proto: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event_obj);
            cmd_buf.encodeSignalEvent_value(proto, value);
            Ok(())
        })
    }

    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        let (event_obj, value) = {
            let slab = self.events.lock();
            let idx = (event as usize)
                .checked_sub(1)
                .ok_or_else(|| anyhow!("stream_wait_event: invalid event handle {event}"))?;
            let slot = slab
                .get(idx)
                .ok_or_else(|| anyhow!("stream_wait_event: event handle {event} out of range"))?;
            // Wait on the most-recently-recorded value (next - 1).
            // If nothing has been recorded yet, slot.next is 1 and
            // value is 0 — Metal treats wait-for-0 as a no-op.
            (slot.event.clone(), slot.next.saturating_sub(1))
        };
        self.with_stream(stream, |s| {
            s.end_encoder();
            let cmd_buf = s.cmd_buf()?;
            let proto: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event_obj);
            cmd_buf.encodeWaitForEvent_value(proto, value);
            Ok(())
        })
    }

    fn destroy_event(&self, event: u64) -> Result<()> {
        if event == 0 {
            return Ok(());
        }
        let mut slab = self.events.lock();
        let idx = (event - 1) as usize;
        if let Some(slot) = slab.get_mut(idx) {
            // Replace with a fresh dummy event so the slab indices
            // stay stable across destroys (matches the cuda backend
            // semantics — handles are not reused).
            slot.next = 0;
        }
        Ok(())
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        // UMA: a Shared MTLBuffer's contents() pointer IS host-pinned
        // memory from the GPU's perspective. We park the buffer in
        // the alloc table keyed by gpuAddress, then return the host
        // pointer. `free_host_pinned` looks the buffer up by host
        // pointer to release it.
        let buf = self
            .device
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("alloc_host_pinned: newBufferWithLength failed"))?;
        let host_ptr = buf.contents().as_ptr() as *mut u8;
        // Stash by gpuAddress so plain `free()` on the DevicePtr would
        // also work; the host-pinned variant is purely a CPU view.
        let addr = buf.gpuAddress();
        if addr == 0 {
            bail!("alloc_host_pinned: gpuAddress returned 0");
        }
        self.allocations.lock().insert(addr, buf);
        Ok(host_ptr)
    }

    fn free_host_pinned(&self, ptr: *mut u8, _bytes: usize) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        // Find the buffer whose contents() pointer matches.
        let mut allocs = self.allocations.lock();
        let target_addr = allocs.iter().find_map(|(addr, buf)| {
            let host = buf.contents().as_ptr() as *mut u8;
            if host == ptr { Some(*addr) } else { None }
        });
        if let Some(addr) = target_addr {
            allocs.remove(&addr);
        }
        Ok(())
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// `ATLAS_METAL_PROFILE=1` commits + waits after EVERY `launch_typed`
/// and attributes each command buffer's GPU time to its kernel name.
/// Massive serialization overhead — for finding where GPU time goes,
/// never for serving. Dumps a sorted table every 8192 launches.
fn profile_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("ATLAS_METAL_PROFILE").is_some())
}

fn profile_record(name: &str, busy_s: f64) {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Mutex<HashMap<String, (u64, f64)>>> = OnceLock::new();
    static SINCE_DUMP: OnceLock<Mutex<u64>> = OnceLock::new();
    let table = TABLE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let mut t = table.lock();
        let e = t.entry(name.to_string()).or_insert((0, 0.0));
        e.0 += 1;
        if busy_s.is_finite() && busy_s > 0.0 {
            e.1 += busy_s;
        }
    }
    let mut n = SINCE_DUMP.get_or_init(|| Mutex::new(0)).lock();
    *n += 1;
    if *n >= 8192 {
        *n = 0;
        let mut rows: Vec<(String, (u64, f64))> =
            table.lock().iter().map(|(k, v)| (k.clone(), *v)).collect();
        rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
        let total: f64 = rows.iter().map(|r| r.1.1).sum();
        let mut msg = format!("metal profile (cumulative GPU time, total {:.1} ms):", total * 1e3);
        for (name, (count, secs)) in rows.iter().take(16) {
            msg.push_str(&format!(
                "\n  {:<32} {:>8} calls {:>9.1} ms ({:>4.1}%)",
                name,
                count,
                secs * 1e3,
                100.0 * secs / total
            ));
        }
        tracing::info!("{msg}");
    }
}

/// `ATLAS_METAL_CB_TIMING=1` accumulates per-command-buffer GPU busy
/// time (GPUEndTime − GPUStartTime) and logs a window summary every 256
/// completed buffers. Splits wall time into GPU-busy vs host overhead
/// without an Instruments trace.
fn record_cb_timing(cb: &ProtocolObject<dyn MTLCommandBuffer>) {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os("ATLAS_METAL_CB_TIMING").is_some()) {
        return;
    }
    static BUSY_NS: AtomicU64 = AtomicU64::new(0);
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let busy_s = cb.GPUEndTime() - cb.GPUStartTime();
    if busy_s.is_finite() && busy_s > 0.0 {
        BUSY_NS.fetch_add((busy_s * 1e9) as u64, Ordering::Relaxed);
    }
    let n = COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_multiple_of(256) {
        let busy_ms = BUSY_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
        tracing::info!(
            "metal cb timing: {busy_ms:.1} ms GPU-busy across last 256 command buffers \
             ({:.3} ms avg)",
            busy_ms / 256.0
        );
    }
}

/// Probe `hw.memsize` via libc::sysctl on macOS. Returns the total
/// system RAM in bytes — on Apple Silicon UMA this is also the upper
/// bound on Metal-addressable memory.
fn sysctl_memsize() -> Option<usize> {
    use std::ffi::CString;
    let name = CString::new("hw.memsize").ok()?;
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut value as *mut u64 as *mut c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 { Some(value as usize) } else { None }
}

// ── Smoke test ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
