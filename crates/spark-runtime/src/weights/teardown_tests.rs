// SPDX-License-Identifier: AGPL-3.0-only

//! `WeightStore` teardown + FP8 KV-scale-count tests — hoisted from
//! `weights.rs` to keep it under the 500 LoC cap.

use super::*;
use crate::gpu::mock::MockGpuBackend;
use atlas_core::scope::{ModelResource, Teardown};
use std::collections::HashMap;

fn store_with(gpu: &dyn GpuBackend, n: usize) -> WeightStore {
    let mut map = HashMap::new();
    for i in 0..n {
        map.insert(
            format!("w{i}"),
            WeightTensor {
                ptr: gpu.alloc(1024).expect("alloc"),
                shape: vec![16, 16],
                dtype: WeightDtype::BF16,
            },
        );
    }
    WeightStore::from_map(map)
}

#[test]
fn releasing_frees_every_tensor() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 8);
    assert_eq!(gpu.alloc_count(), 8);
    store.release(&gpu).expect("released");
    assert_eq!(gpu.alloc_count(), 0, "every weight was freed");
    assert_eq!(store.len(), 0, "and the map does not hold dead pointers");
}

/// The contract says idempotent: the host calls it, and a `Drop` backstop
/// may call it again. A second call must not double-free.
#[test]
fn releasing_twice_is_harmless() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 4);
    store.release(&gpu).expect("first");
    store.release(&gpu).expect("second");
    assert_eq!(gpu.alloc_count(), 0);
}

/// `fp8_kv_scale_count` counts exactly the `*.k_scale` tensors — one per
/// attention layer in checkpoints that ship calibrated FP8 KV scales —
/// and ignores `v_scale` (paired 1:1 with `k_scale`, counting both would
/// double-report) and lookalike suffixes.
#[test]
fn fp8_kv_scale_count_counts_only_k_scale_tensors() {
    let gpu = MockGpuBackend::new();
    let tensor = || WeightTensor {
        ptr: gpu.alloc(1024).expect("alloc"),
        shape: vec![1],
        dtype: WeightDtype::BF16,
    };
    let mut map = HashMap::new();
    for name in [
        "model.layers.0.self_attn.k_scale",
        "model.layers.0.self_attn.v_scale",
        "model.layers.7.self_attn.k_scale",
        "model.layers.7.self_attn.v_scale",
        "model.layers.0.self_attn.q_proj.weight",
        // Lookalikes that must NOT count: no dot before the suffix, and a
        // different scale kind entirely.
        "model.layers.0.self_attn.attnk_scale",
        "model.layers.0.mlp.weight_scale",
    ] {
        map.insert(name.to_string(), tensor());
    }
    let store = WeightStore::from_map(map);
    assert_eq!(store.fp8_kv_scale_count(), 2);
}

/// A checkpoint without shipped KV scales reports zero — the case where
/// serve logs the "needs calibration or a non-FP8 KV dtype" warning.
#[test]
fn fp8_kv_scale_count_zero_without_scales() {
    let gpu = MockGpuBackend::new();
    let store = store_with(&gpu, 4);
    assert_eq!(store.fp8_kv_scale_count(), 0);
}

fn tensor(gpu: &dyn GpuBackend, dtype: WeightDtype, shape: &[usize]) -> WeightTensor {
    let n: usize = shape.iter().product();
    WeightTensor {
        ptr: gpu.alloc(n * dtype.byte_size().max(1)).expect("alloc"),
        shape: shape.to_vec(),
        dtype,
    }
}

/// The pointer test IS the mechanism. `dense_auto` hands back the STORE
/// POINTER for a BF16 tensor and a FRESH allocation for an FP8 one; only the
/// second is the caller's to free, and freeing the first would hand the store
/// a dangling pointer.
#[test]
fn retire_dequant_source_takes_the_fresh_buffer_and_spares_the_alias() {
    let gpu = MockGpuBackend::new();
    let mut map = HashMap::new();
    // 32 elements each: FP8 on disk (dequants to a fresh 64-byte BF16), BF16
    // on disk (aliases straight through).
    map.insert(
        "a.weight".to_string(),
        tensor(&gpu, WeightDtype::FP8E4M3, &[4, 8]),
    );
    map.insert(
        "b.weight".to_string(),
        tensor(&gpu, WeightDtype::BF16, &[4, 8]),
    );
    let alias = map["b.weight"].ptr;
    let mut store = WeightStore::from_map(map);

    let dequant = gpu.alloc(64).expect("alloc");
    assert!(store.retire_dequant_source("a", dequant), "fresh → queued");
    assert!(
        !store.retire_dequant_source("b", alias),
        "store alias → not ours"
    );
    assert!(
        !store.retire_dequant_source("nope", dequant),
        "unknown prefix → nothing to compare against"
    );
    assert!(!store.retire_dequant_source("a", DevicePtr::NULL));

    assert_eq!(gpu.alloc_count(), 3);
    let (bytes, count) = store.free_retired(&gpu).expect("freed");
    assert_eq!((bytes, count), (64, 1));
    assert_eq!(gpu.alloc_count(), 2, "only the dequant went");
    assert!(store.get("b.weight").is_ok(), "the alias is still live");
}

/// A 4-bit-packed source is `[n, k/2]` on disk, so its element count is HALF
/// the BF16 output's. Getting this wrong would poison (or, on a real backend,
/// memset) past the end of the buffer.
#[test]
fn retire_dequant_source_sizes_a_packed_source_from_the_unpacked_width() {
    let gpu = MockGpuBackend::new();
    let mut map = HashMap::new();
    // [4, 4] U8 = 16 bytes on disk = 32 logical values = 64 BF16 bytes.
    map.insert(
        "c.weight_packed".to_string(),
        tensor(&gpu, WeightDtype::UInt8, &[4, 4]),
    );
    let mut store = WeightStore::from_map(map);
    assert!(store.retire_dequant_source("c", gpu.alloc(64).expect("alloc")));
    assert_eq!(store.free_retired(&gpu).expect("freed"), (64, 1));
}

/// `free_retired` is the ordinary path, but a load that errors out before it
/// must not strand the buffers — teardown is the backstop, and these are not
/// store tensors so `weights.drain()` cannot have covered them.
#[test]
fn release_frees_a_dequant_free_retired_never_took() {
    let gpu = MockGpuBackend::new();
    let mut map = HashMap::new();
    map.insert(
        "a.weight".to_string(),
        tensor(&gpu, WeightDtype::FP8E4M3, &[4, 8]),
    );
    let mut store = WeightStore::from_map(map);
    assert!(store.retire_dequant_source("a", gpu.alloc(64).expect("alloc")));
    assert_eq!(gpu.alloc_count(), 2);
    store.release(&gpu).expect("released");
    assert_eq!(gpu.alloc_count(), 0);
}

/// Reverse order, and one failure does not abandon the rest — the whole
/// reason `Teardown` exists rather than `Drop`.
#[test]
fn teardown_releases_in_reverse_registration_order() {
    let gpu = MockGpuBackend::new();
    let mut teardown: Teardown<dyn GpuBackend> = Teardown::new();
    teardown.push(Box::new(store_with(&gpu, 3)));
    teardown.push(Box::new(store_with(&gpu, 5)));
    assert_eq!(gpu.alloc_count(), 8);
    teardown.release_all(&gpu).expect("released");
    assert_eq!(gpu.alloc_count(), 0);
    assert!(teardown.is_empty());
}
