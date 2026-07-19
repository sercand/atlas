// SPDX-License-Identifier: AGPL-3.0-only
//! Keep-packed PrismML Q1_0 (ggml id 41) GEMV parity — the decode-path
//! matvec the Bonsai-27B Metal model runs every projection through.
//!
//! Fixtures are hand-packed `block_q1_0` bytes (fp16 scale + 128
//! LSB-first sign bits, 18-byte block) validated against the same CPU
//! oracle the GGUF loader uses (`dequant_cpu::dequant_to_f32`), so a
//! layout drift between loader and kernel cannot pass.

#[allow(unused_imports)]
use super::super::*;
#[allow(unused_imports)]
use super::helpers::*;
use crate::gpu::{GpuBackend, KernelArg};
use crate::weights::gguf_q1::{Q1_BLOCK_BYTES, Q1_GROUP, dequant_row_f32};

/// Deterministic synthetic Q1_0 weight: `(packed_bytes, w_dequant_f32)`
/// for an `[n_rows, n_cols]` row-major weight (`n_cols % 128 == 0`).
fn build_q1_fixture(n_rows: usize, n_cols: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(n_cols % Q1_GROUP, 0);
    let blocks_per_row = n_cols / Q1_GROUP;
    let mut packed = Vec::with_capacity(n_rows * blocks_per_row * Q1_BLOCK_BYTES);
    for r in 0..n_rows {
        for b in 0..blocks_per_row {
            // Per-block fp16 scale, kept small like real 1-bit checkpoints.
            let d = half::f16::from_f32(0.005 + 0.002 * r as f32 + 0.0007 * b as f32);
            packed.extend_from_slice(&d.to_le_bytes());
            // 16 sign-bit bytes, deterministic pattern spanning all bits.
            for byte in 0..16 {
                packed.push(((r * 31 + b * 17 + byte * 97 + 13) % 256) as u8);
            }
        }
    }
    // Reference dequant through the canonical CPU implementation (the
    // same one the GGUF loader's oracle delegates to).
    let mut w = vec![0f32; n_rows * n_cols];
    for r in 0..n_rows {
        let row_bytes = &packed[r * blocks_per_row * Q1_BLOCK_BYTES..];
        dequant_row_f32(
            &row_bytes[..blocks_per_row * Q1_BLOCK_BYTES],
            n_cols,
            &mut w[r * n_cols..(r + 1) * n_cols],
        )
        .expect("cpu q1_0 dequant");
    }
    (packed, w)
}

#[test]
fn metal_q1_0_gemv_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let Ok(kernel) = backend.kernel("q1_0_gemv", "q1_0_gemv") else {
        eprintln!("skipping: q1_0_gemv not in this kernel build (set ATLAS_TARGET_MODEL)");
        return;
    };

    // 2 threadgroups' worth of rows, 3 blocks per row.
    let n = 8u32;
    let k = 384u32;
    let (packed, w) = build_q1_fixture(n as usize, k as usize);

    let x: Vec<half::bf16> = (0..k as usize)
        .map(|i| half::bf16::from_f32(test_val(i)))
        .collect();

    // FP32 reference accumulation from the BF16-rounded x.
    let mut expected = vec![0f32; n as usize];
    for r in 0..n as usize {
        let mut acc = 0f32;
        for c in 0..k as usize {
            acc += w[r * k as usize + c] * x[c].to_f32();
        }
        expected[r] = acc;
    }

    let packed_ptr = backend.alloc(packed.len()).expect("alloc packed");
    let x_ptr = backend.alloc(x.len() * 2).expect("alloc x");
    let y_ptr = backend.alloc(n as usize * 2).expect("alloc y");
    backend.copy_h2d(&packed, packed_ptr).expect("h2d packed");
    backend
        .copy_h2d(&bf16_slice_to_bytes(&x), x_ptr)
        .expect("h2d x");

    backend
        .launch_typed(
            kernel,
            [n.div_ceil(4), 1, 1],
            [128, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Bytes(&k.to_le_bytes()),
                KernelArg::Buffer(packed_ptr),
                KernelArg::Buffer(x_ptr),
                KernelArg::Buffer(y_ptr),
            ],
        )
        .expect("launch q1_0_gemv");
    backend
        .synchronize(backend.default_stream())
        .expect("synchronize");

    let mut y_raw = vec![0u8; n as usize * 2];
    backend.copy_d2h(y_ptr, &mut y_raw).expect("d2h y");
    let actual = bytes_to_bf16_vec(&y_raw);

    for r in 0..n as usize {
        let e = expected[r];
        let a = actual[r].to_f32();
        // BF16 output rounding + FP32 tree-vs-serial reduction order.
        let tol = (e.abs() * 1e-2).max(2e-3);
        assert!(
            (e - a).abs() <= tol,
            "row {r}: expected {e}, got {a} (tol {tol})"
        );
    }
}

#[test]
fn metal_q1_0_gemv_batchm_matches_single_rows() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let Ok(kernel) = backend.kernel("q1_0_gemv", "q1_0_gemv_batchm") else {
        eprintln!("skipping: q1_0_gemv not in this kernel build (set ATLAS_TARGET_MODEL)");
        return;
    };

    let n = 6u32;
    let k = 256u32;
    let m = 3u32;
    let (packed, w) = build_q1_fixture(n as usize, k as usize);

    let x: Vec<half::bf16> = (0..(m * k) as usize)
        .map(|i| half::bf16::from_f32(test_val(i * 3 + 1)))
        .collect();

    let mut expected = vec![0f32; (m * n) as usize];
    for lane in 0..m as usize {
        for r in 0..n as usize {
            let mut acc = 0f32;
            for c in 0..k as usize {
                acc += w[r * k as usize + c] * x[lane * k as usize + c].to_f32();
            }
            expected[lane * n as usize + r] = acc;
        }
    }

    let packed_ptr = backend.alloc(packed.len()).expect("alloc packed");
    let x_ptr = backend.alloc(x.len() * 2).expect("alloc x");
    let y_ptr = backend.alloc((m * n) as usize * 2).expect("alloc y");
    backend.copy_h2d(&packed, packed_ptr).expect("h2d packed");
    backend
        .copy_h2d(&bf16_slice_to_bytes(&x), x_ptr)
        .expect("h2d x");

    backend
        .launch_typed(
            kernel,
            [n.div_ceil(4), 1, 1],
            [128, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Bytes(&k.to_le_bytes()),
                KernelArg::Bytes(&m.to_le_bytes()),
                KernelArg::Buffer(packed_ptr),
                KernelArg::Buffer(x_ptr),
                KernelArg::Buffer(y_ptr),
            ],
        )
        .expect("launch q1_0_gemv_batchm");
    backend
        .synchronize(backend.default_stream())
        .expect("synchronize");

    let mut y_raw = vec![0u8; (m * n) as usize * 2];
    backend.copy_d2h(y_ptr, &mut y_raw).expect("d2h y");
    let actual = bytes_to_bf16_vec(&y_raw);

    for i in 0..(m * n) as usize {
        let e = expected[i];
        let a = actual[i].to_f32();
        let tol = (e.abs() * 1e-2).max(2e-3);
        assert!(
            (e - a).abs() <= tol,
            "y[{i}]: expected {e}, got {a} (tol {tol})"
        );
    }
}
