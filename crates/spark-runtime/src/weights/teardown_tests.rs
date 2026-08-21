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

// ── retirement ────────────────────────────────────────────────────────────
//
// `retire` + `free_retired` release store tensors whose consumers all copied
// out of them, so a derived-copy-heavy checkpoint does not keep the original
// resident for the model's lifetime. The tests below pin the safety property
// that makes it usable: a tensor some consumer ALIASED is never freed.

#[test]
fn retiring_frees_only_the_retired_tensor() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 4);
    assert_eq!(gpu.alloc_count(), 4);

    // A derived copy, i.e. a pointer that is not the store's.
    let copy = gpu.alloc(1024).expect("alloc");
    assert!(store.retire("w0", &[copy]));

    let (count, bytes) = store.free_retired(&gpu).expect("free");
    assert_eq!(count, 1);
    assert_eq!(bytes, 16 * 16 * 2, "BF16 [16,16]");
    assert_eq!(gpu.alloc_count(), 4, "3 weights + the derived copy remain");
    assert!(store.get("w0").is_err(), "retired tensor left the map");
    assert!(store.get("w1").is_ok(), "others untouched");
}

/// THE safety property. `dense_auto` returns the store's own pointer for BF16
/// tensors and a fresh buffer for FP8, so whether a call site copied depends on
/// the checkpoint, not on the code. Passing the derived pointer lets `retire`
/// settle that at runtime, and it must refuse when they are the same pointer.
#[test]
fn retiring_a_tensor_a_consumer_aliased_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 2);
    let aliased = store.get("w0").expect("w0").ptr;

    assert!(
        !store.retire("w0", &[aliased]),
        "a consumer holding the store pointer must block retirement"
    );

    let (count, _) = store.free_retired(&gpu).expect("free");
    assert_eq!(count, 0);
    assert_eq!(gpu.alloc_count(), 2, "nothing was freed");
    assert!(store.get("w0").is_ok(), "and it is still readable");
}

/// Retiring then releasing must not double-free: `free_retired` removes the
/// entry, so `release` never sees it.
#[test]
fn retire_then_release_does_not_double_free() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 3);
    let copy = gpu.alloc(1024).expect("alloc");
    assert!(store.retire("w1", &[copy]));
    store.free_retired(&gpu).expect("free retired");
    store.release(&gpu).expect("release");
    assert_eq!(gpu.alloc_count(), 1, "only the unrelated derived copy is left");
}

#[test]
fn free_retired_is_idempotent() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 2);
    let copy = gpu.alloc(1024).expect("alloc");
    store.retire("w0", &[copy]);
    assert_eq!(store.free_retired(&gpu).expect("first").0, 1);
    assert_eq!(store.free_retired(&gpu).expect("second").0, 0);
}

/// Retiring a name that is not in the store is a no-op, not an error — the
/// loaders retire by constructed name and a checkpoint need not carry it.
#[test]
fn retiring_an_absent_tensor_is_a_noop() {
    let gpu = MockGpuBackend::new();
    let store = store_with(&gpu, 1);
    assert!(!store.retire("does.not.exist", &[]));
}
