// SPDX-License-Identifier: AGPL-3.0-only
//! Batched-prefill kernel parity: `q1_0_gemm` against a CPU dequant
//! reference, and the chunked GDN / conv1d prefill kernels against a
//! token-by-token walk of their trusted DECODE counterparts (same
//! state, same inputs — the prefill kernels must reproduce the decode
//! recurrence, not just some plausible math).

#[allow(unused_imports)]
use super::super::*;
#[allow(unused_imports)]
use super::helpers::*;
use crate::gpu::{GpuBackend, KernelArg};
use crate::weights::gguf_q1::{Q1_BLOCK_BYTES, Q1_GROUP, dequant_row_f32};

fn test_val(i: usize) -> f32 {
    ((i * 37 + 11) % 41) as f32 * 0.03 - 0.6
}

fn build_q1_fixture(n_rows: usize, n_cols: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(n_cols % Q1_GROUP, 0);
    let blocks_per_row = n_cols / Q1_GROUP;
    let mut packed = Vec::with_capacity(n_rows * blocks_per_row * Q1_BLOCK_BYTES);
    for r in 0..n_rows {
        for b in 0..blocks_per_row {
            let d = half::f16::from_f32(0.005 + 0.002 * r as f32 + 0.0007 * b as f32);
            packed.extend_from_slice(&d.to_le_bytes());
            for byte in 0..16 {
                packed.push(((r * 31 + b * 17 + byte * 97 + 13) % 256) as u8);
            }
        }
    }
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
fn metal_q1_0_gemm_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let Ok(kernel) = backend.kernel("q1_0_gemm", "q1_0_gemm") else {
        eprintln!("skipping: q1_0_gemm not in this kernel build");
        return;
    };

    // Tail-exercising shape: 40 rows (1¼ row tiles), 45 tokens (1½
    // token tiles), 3 blocks per row.
    let n = 40u32;
    let k = 384u32;
    let t = 45u32;
    let (packed, w) = build_q1_fixture(n as usize, k as usize);

    let x: Vec<half::bf16> = (0..(t * k) as usize)
        .map(|i| half::bf16::from_f32(test_val(i)))
        .collect();

    let mut expected = vec![0f32; (t * n) as usize];
    for tok in 0..t as usize {
        for r in 0..n as usize {
            let mut acc = 0f32;
            for c in 0..k as usize {
                acc += w[r * k as usize + c] * x[tok * k as usize + c].to_f32();
            }
            expected[tok * n as usize + r] = acc;
        }
    }

    // The kernel reads x as HALF with rows padded to the 128-token
    // tile (caller contract) — build the padded half image directly.
    let t_pad = t.next_multiple_of(128);
    let mut x_half_bytes = vec![0u8; (t_pad * k) as usize * 2];
    for (i, v) in x.iter().enumerate() {
        x_half_bytes[i * 2..i * 2 + 2]
            .copy_from_slice(&half::f16::from_f32(v.to_f32()).to_le_bytes());
    }

    let packed_ptr = backend.alloc(packed.len()).expect("alloc packed");
    let x_ptr = backend.alloc(x_half_bytes.len()).expect("alloc x");
    let y_ptr = backend.alloc((t * n) as usize * 2).expect("alloc y");
    backend.copy_h2d(&packed, packed_ptr).expect("h2d packed");
    backend.copy_h2d(&x_half_bytes, x_ptr).expect("h2d x");

    backend
        .launch_typed(
            kernel,
            [n.div_ceil(32) * t.div_ceil(128), 1, 1],
            [128, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Bytes(&k.to_le_bytes()),
                KernelArg::Bytes(&t.to_le_bytes()),
                KernelArg::Buffer(packed_ptr),
                KernelArg::Buffer(x_ptr),
                KernelArg::Buffer(y_ptr),
            ],
        )
        .expect("launch q1_0_gemm");
    backend
        .synchronize(backend.default_stream())
        .expect("synchronize");

    let mut y_raw = vec![0u8; (t * n) as usize * 2];
    backend.copy_d2h(y_ptr, &mut y_raw).expect("d2h y");
    let actual = bytes_to_bf16_vec(&y_raw);

    for i in 0..(t * n) as usize {
        let e = expected[i];
        let a = actual[i].to_f32();
        let tol = (e.abs() * 1e-2).max(2e-3);
        assert!(
            (e - a).abs() <= tol,
            "y[{i}] (tok {}, row {}): expected {e}, got {a} (tol {tol})",
            i / n as usize,
            i % n as usize
        );
    }
}

/// The chunked GDN prefill kernel must reproduce a token-by-token walk
/// of `gated_delta_rule_decode` — outputs AND final state.
#[test]
fn metal_gdn_prefill_matches_decode_walk() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let Ok(prefill) = backend.kernel("gated_delta_rule_prefill", "gated_delta_rule_prefill") else {
        eprintln!("skipping: gated_delta_rule_prefill not in this kernel build");
        return;
    };
    let decode = backend
        .kernel("gated_delta_rule_decode", "gated_delta_rule_decode")
        .expect("decode kernel");

    let dim = 128usize; // k_dim == v_dim, fixed by the prefill kernel
    let num_k_heads = 2u32;
    let num_v_heads = 6u32;
    let t = 5usize;
    let qkv_stride = (2 * num_k_heads as usize + num_v_heads as usize) * dim;

    // Staged rows: [t, qkv_stride] bf16 (Q | K | V), plus per-token
    // fp32 gate/beta in (0, 1).
    let qkv: Vec<half::bf16> = (0..t * qkv_stride)
        .map(|i| half::bf16::from_f32(test_val(i) * 0.5))
        .collect();
    let gate: Vec<f32> = (0..t * num_v_heads as usize)
        .map(|i| 0.55 + 0.4 * ((i % 7) as f32 / 7.0))
        .collect();
    let beta: Vec<f32> = (0..t * num_v_heads as usize)
        .map(|i| 0.3 + 0.6 * ((i % 5) as f32 / 5.0))
        .collect();
    let state0: Vec<f32> = (0..num_v_heads as usize * dim * dim)
        .map(|i| test_val(i * 7) * 0.02)
        .collect();

    let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };

    let qkv_ptr = backend.alloc(qkv.len() * 2).expect("alloc qkv");
    backend
        .copy_h2d(&bf16_slice_to_bytes(&qkv), qkv_ptr)
        .expect("h2d qkv");
    let gate_ptr = backend.alloc(gate.len() * 4).expect("alloc gate");
    backend.copy_h2d(&f32_bytes(&gate), gate_ptr).expect("h2d");
    let beta_ptr = backend.alloc(beta.len() * 4).expect("alloc beta");
    backend.copy_h2d(&f32_bytes(&beta), beta_ptr).expect("h2d");

    let state_ref = backend.alloc(state0.len() * 4).expect("alloc state");
    let state_new = backend.alloc(state0.len() * 4).expect("alloc state");
    backend
        .copy_h2d(&f32_bytes(&state0), state_ref)
        .expect("h2d state");
    backend
        .copy_h2d(&f32_bytes(&state0), state_new)
        .expect("h2d state");

    let z_dim = num_v_heads as usize * dim;
    let y_ref = backend.alloc(t * z_dim * 2).expect("alloc y_ref");
    let y_new = backend.alloc(t * z_dim * 2).expect("alloc y_new");

    let stream = backend.default_stream();
    let batch_one = 1u32;
    let dims = [dim as u32, dim as u32];
    // Reference: T sequential decode dispatches over the same rows.
    for tok in 0..t {
        let row = qkv_ptr.offset(tok * qkv_stride * 2);
        let k_off = num_k_heads as usize * dim * 2;
        let v_off = 2 * num_k_heads as usize * dim * 2;
        backend
            .launch_typed(
                decode,
                [num_v_heads, 1, 1],
                [128, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Buffer(state_ref),
                    KernelArg::Buffer(row),
                    KernelArg::Buffer(row.offset(k_off)),
                    KernelArg::Buffer(row.offset(v_off)),
                    KernelArg::Buffer(gate_ptr.offset(tok * num_v_heads as usize * 4)),
                    KernelArg::Buffer(beta_ptr.offset(tok * num_v_heads as usize * 4)),
                    KernelArg::Buffer(y_ref.offset(tok * z_dim * 2)),
                    KernelArg::Bytes(&batch_one.to_le_bytes()),
                    KernelArg::Bytes(&num_k_heads.to_le_bytes()),
                    KernelArg::Bytes(&num_v_heads.to_le_bytes()),
                    KernelArg::Bytes(&dims[0].to_le_bytes()),
                    KernelArg::Bytes(&dims[1].to_le_bytes()),
                ],
            )
            .expect("launch decode");
    }
    // Chunked prefill in one dispatch.
    let t_u32 = t as u32;
    let stride_u32 = qkv_stride as u32;
    backend
        .launch_typed(
            prefill,
            [num_v_heads, 1, 1],
            [128, 1, 1],
            0,
            stream,
            &[
                KernelArg::Buffer(state_new),
                KernelArg::Buffer(qkv_ptr),
                KernelArg::Buffer(gate_ptr),
                KernelArg::Buffer(beta_ptr),
                KernelArg::Buffer(y_new),
                KernelArg::Bytes(&t_u32.to_le_bytes()),
                KernelArg::Bytes(&num_k_heads.to_le_bytes()),
                KernelArg::Bytes(&num_v_heads.to_le_bytes()),
                KernelArg::Bytes(&stride_u32.to_le_bytes()),
            ],
        )
        .expect("launch prefill");
    backend.synchronize(stream).expect("synchronize");

    let mut yr = vec![0u8; t * z_dim * 2];
    let mut yn = vec![0u8; t * z_dim * 2];
    backend.copy_d2h(y_ref, &mut yr).expect("d2h");
    backend.copy_d2h(y_new, &mut yn).expect("d2h");
    let yr = bytes_to_bf16_vec(&yr);
    let yn = bytes_to_bf16_vec(&yn);
    for i in 0..t * z_dim {
        let e = yr[i].to_f32();
        let a = yn[i].to_f32();
        let tol = (e.abs() * 1e-2).max(3e-3);
        assert!(
            (e - a).abs() <= tol,
            "y[{i}] (tok {}): decode-walk {e}, prefill {a}",
            i / z_dim
        );
    }

    let mut sr = vec![0u8; state0.len() * 4];
    let mut sn = vec![0u8; state0.len() * 4];
    backend.copy_d2h(state_ref, &mut sr).expect("d2h");
    backend.copy_d2h(state_new, &mut sn).expect("d2h");
    for i in 0..state0.len() {
        let e = f32::from_le_bytes(sr[i * 4..i * 4 + 4].try_into().unwrap());
        let a = f32::from_le_bytes(sn[i * 4..i * 4 + 4].try_into().unwrap());
        let tol = (e.abs() * 1e-3).max(1e-4);
        assert!(
            (e - a).abs() <= tol,
            "state[{i}]: decode-walk {e}, prefill {a}"
        );
    }
}

/// Conv1d prefill (+state advance) vs a token-by-token decode walk.
#[test]
fn metal_conv1d_prefill_matches_decode_walk() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let Ok(prefill) = backend.kernel("causal_conv1d_prefill", "causal_conv1d_prefill_l2norm")
    else {
        eprintln!("skipping: causal_conv1d_prefill not in this kernel build");
        return;
    };
    let advance = backend
        .kernel("causal_conv1d_prefill", "causal_conv1d_prefill_state")
        .expect("state kernel");
    let decode = backend
        .kernel(
            "causal_conv1d_update_l2norm",
            "causal_conv1d_update_l2norm",
        )
        .expect("decode kernel");

    let head_dim = 128u32;
    let dim = 384u32; // 2 L2-normed heads + one plain-SiLU (V) block
    let qk_channels = 256u32;
    let d_conv = 4u32;
    let t = 6usize;
    let l2_eps = 1e-6f32;

    let input: Vec<half::bf16> = (0..t * dim as usize)
        .map(|i| half::bf16::from_f32(test_val(i * 5 + 3)))
        .collect();
    let weight: Vec<half::bf16> = (0..(dim * d_conv) as usize)
        .map(|i| half::bf16::from_f32(test_val(i * 11) * 0.4))
        .collect();
    let state0: Vec<f32> = (0..(dim * d_conv) as usize)
        .map(|i| test_val(i * 13) * 0.3)
        .collect();
    let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };

    let in_ptr = backend.alloc(input.len() * 2).expect("alloc in");
    backend
        .copy_h2d(&bf16_slice_to_bytes(&input), in_ptr)
        .expect("h2d");
    let w_ptr = backend.alloc(weight.len() * 2).expect("alloc w");
    backend
        .copy_h2d(&bf16_slice_to_bytes(&weight), w_ptr)
        .expect("h2d");
    let state_ref = backend.alloc(state0.len() * 4).expect("alloc");
    let state_new = backend.alloc(state0.len() * 4).expect("alloc");
    backend.copy_h2d(&f32_bytes(&state0), state_ref).expect("h2d");
    backend.copy_h2d(&f32_bytes(&state0), state_new).expect("h2d");
    let out_ref = backend.alloc(t * dim as usize * 2).expect("alloc");
    let out_new = backend.alloc(t * dim as usize * 2).expect("alloc");

    let stream = backend.default_stream();
    let batch_one = 1u32;
    // Reference walk: decode kernel once per token (state evolves).
    for tok in 0..t {
        backend
            .launch_typed(
                decode,
                [dim.div_ceil(head_dim) * batch_one, 1, 1],
                [head_dim, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Buffer(state_ref),
                    KernelArg::Buffer(in_ptr.offset(tok * dim as usize * 2)),
                    KernelArg::Buffer(w_ptr),
                    KernelArg::Buffer(out_ref.offset(tok * dim as usize * 2)),
                    KernelArg::Bytes(&batch_one.to_le_bytes()),
                    KernelArg::Bytes(&dim.to_le_bytes()),
                    KernelArg::Bytes(&d_conv.to_le_bytes()),
                    KernelArg::Bytes(&qk_channels.to_le_bytes()),
                    KernelArg::Bytes(&head_dim.to_le_bytes()),
                    KernelArg::Bytes(&l2_eps.to_le_bytes()),
                ],
            )
            .expect("launch decode conv");
    }
    // Batched prefill + state advance.
    let t_u32 = t as u32;
    backend
        .launch_typed(
            prefill,
            [dim.div_ceil(head_dim) * t_u32, 1, 1],
            [head_dim, 1, 1],
            0,
            stream,
            &[
                KernelArg::Buffer(state_new),
                KernelArg::Buffer(in_ptr),
                KernelArg::Buffer(w_ptr),
                KernelArg::Buffer(out_new),
                KernelArg::Bytes(&t_u32.to_le_bytes()),
                KernelArg::Bytes(&dim.to_le_bytes()),
                KernelArg::Bytes(&d_conv.to_le_bytes()),
                KernelArg::Bytes(&qk_channels.to_le_bytes()),
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&l2_eps.to_le_bytes()),
            ],
        )
        .expect("launch prefill conv");
    backend
        .launch_typed(
            advance,
            [dim.div_ceil(128), 1, 1],
            [128, 1, 1],
            0,
            stream,
            &[
                KernelArg::Buffer(state_new),
                KernelArg::Buffer(in_ptr),
                KernelArg::Bytes(&t_u32.to_le_bytes()),
                KernelArg::Bytes(&dim.to_le_bytes()),
                KernelArg::Bytes(&d_conv.to_le_bytes()),
            ],
        )
        .expect("launch state advance");
    backend.synchronize(stream).expect("synchronize");

    let mut or_ = vec![0u8; t * dim as usize * 2];
    let mut on = vec![0u8; t * dim as usize * 2];
    backend.copy_d2h(out_ref, &mut or_).expect("d2h");
    backend.copy_d2h(out_new, &mut on).expect("d2h");
    let or_ = bytes_to_bf16_vec(&or_);
    let on = bytes_to_bf16_vec(&on);
    for i in 0..t * dim as usize {
        let e = or_[i].to_f32();
        let a = on[i].to_f32();
        let tol = (e.abs() * 1e-2).max(2e-3);
        assert!(
            (e - a).abs() <= tol,
            "out[{i}] (tok {}, ch {}): decode-walk {e}, prefill {a}",
            i / dim as usize,
            i % dim as usize
        );
    }
    let mut sr = vec![0u8; state0.len() * 4];
    let mut sn = vec![0u8; state0.len() * 4];
    backend.copy_d2h(state_ref, &mut sr).expect("d2h");
    backend.copy_d2h(state_new, &mut sn).expect("d2h");
    for i in 0..state0.len() {
        let e = f32::from_le_bytes(sr[i * 4..i * 4 + 4].try_into().unwrap());
        let a = f32::from_le_bytes(sn[i * 4..i * 4 + 4].try_into().unwrap());
        assert!(
            (e - a).abs() <= 1e-4,
            "conv state[{i}]: decode-walk {e}, prefill {a}"
        );
    }
}

/// `attention_prefill_offset` against a CPU causal-softmax reference
/// over a KV history longer than the query block (chunk_start > 0).
#[test]
fn metal_attention_prefill_offset_matches_cpu() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let Ok(kernel) = backend.kernel("attention_prefill", "attention_prefill_offset") else {
        eprintln!("skipping: attention_prefill_offset not in this kernel build");
        return;
    };

    let num_heads = 4u32;
    let num_kv_heads = 2u32;
    let head_dim = 16u32;
    let pos_base = 3u32;
    let t = 5u32;
    let seq_len = pos_base + t;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q: Vec<half::bf16> = (0..(t * num_heads * head_dim) as usize)
        .map(|i| half::bf16::from_f32(test_val(i)))
        .collect();
    let k: Vec<half::bf16> = (0..(seq_len * num_kv_heads * head_dim) as usize)
        .map(|i| half::bf16::from_f32(test_val(i * 3 + 1)))
        .collect();
    let v: Vec<half::bf16> = (0..(seq_len * num_kv_heads * head_dim) as usize)
        .map(|i| half::bf16::from_f32(test_val(i * 7 + 2)))
        .collect();

    // CPU reference.
    let hd = head_dim as usize;
    let group = (num_heads / num_kv_heads) as usize;
    let mut expected = vec![0f32; (t * num_heads * head_dim) as usize];
    for m in 0..t as usize {
        for h in 0..num_heads as usize {
            let kvh = h / group;
            let cutoff = pos_base as usize + m + 1;
            let mut scores = vec![f32::NEG_INFINITY; seq_len as usize];
            let mut mx = f32::NEG_INFINITY;
            for s in 0..cutoff {
                let mut dot = 0f32;
                for d in 0..hd {
                    dot += q[(m * num_heads as usize + h) * hd + d].to_f32()
                        * k[(s * num_kv_heads as usize + kvh) * hd + d].to_f32();
                }
                scores[s] = dot * scale;
                mx = mx.max(scores[s]);
            }
            let mut sum = 0f32;
            for s in 0..cutoff {
                scores[s] = (scores[s] - mx).exp();
                sum += scores[s];
            }
            for d in 0..hd {
                let mut acc = 0f32;
                for s in 0..cutoff {
                    acc += scores[s] / sum
                        * v[(s * num_kv_heads as usize + kvh) * hd + d].to_f32();
                }
                expected[(m * num_heads as usize + h) * hd + d] = acc;
            }
        }
    }

    let q_ptr = backend.alloc(q.len() * 2).expect("alloc");
    let k_ptr = backend.alloc(k.len() * 2).expect("alloc");
    let v_ptr = backend.alloc(v.len() * 2).expect("alloc");
    let o_ptr = backend.alloc(q.len() * 2).expect("alloc");
    backend.copy_h2d(&bf16_slice_to_bytes(&q), q_ptr).expect("h2d");
    backend.copy_h2d(&bf16_slice_to_bytes(&k), k_ptr).expect("h2d");
    backend.copy_h2d(&bf16_slice_to_bytes(&v), v_ptr).expect("h2d");

    backend
        .launch_typed(
            kernel,
            [t * num_heads, 1, 1],
            [128, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&t.to_le_bytes()),
                KernelArg::Bytes(&seq_len.to_le_bytes()),
                KernelArg::Bytes(&pos_base.to_le_bytes()),
                KernelArg::Bytes(&num_heads.to_le_bytes()),
                KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&scale.to_le_bytes()),
                KernelArg::Buffer(q_ptr),
                KernelArg::Buffer(k_ptr),
                KernelArg::Buffer(v_ptr),
                KernelArg::Buffer(o_ptr),
            ],
        )
        .expect("launch attention_prefill_offset");
    backend
        .synchronize(backend.default_stream())
        .expect("synchronize");

    let mut o_raw = vec![0u8; q.len() * 2];
    backend.copy_d2h(o_ptr, &mut o_raw).expect("d2h");
    let actual = bytes_to_bf16_vec(&o_raw);
    for i in 0..expected.len() {
        let e = expected[i];
        let a = actual[i].to_f32();
        let tol = (e.abs() * 1e-2).max(3e-3);
        assert!((e - a).abs() <= tol, "out[{i}]: expected {e}, got {a}");
    }
}

/// Standalone timing probe for the batched-prefill GEMM at Bonsai FFN
/// dimensions. Not a correctness test — run explicitly with
/// `cargo test ... metal_q1_0_gemm_micro -- --ignored --nocapture`.
#[test]
#[ignore]
fn metal_q1_0_gemm_micro_bench() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let kernel = backend.kernel("q1_0_gemm", "q1_0_gemm").expect("kernel");

    let n = 17408u32;
    let k = 5120u32;
    let t = 256u32;
    let blocks_per_row = (k as usize) / Q1_GROUP;
    let packed_len = n as usize * blocks_per_row * Q1_BLOCK_BYTES;
    // Contents don't matter for timing; allocate zeroed device buffers
    // (x is the pre-cast half image, t is already tile-aligned).
    let packed_ptr = backend.alloc(packed_len).expect("alloc packed");
    let x_ptr = backend.alloc(t as usize * k as usize * 2).expect("alloc x");
    let y_ptr = backend.alloc(t as usize * n as usize * 2).expect("alloc y");

    let stream = backend.default_stream();
    let launch = |iters: u32| {
        let start = std::time::Instant::now();
        for _ in 0..iters {
            backend
                .launch_typed(
                    kernel,
                    [n.div_ceil(32) * t.div_ceil(128), 1, 1],
                    [128, 1, 1],
                    0,
                    stream,
                    &[
                        KernelArg::Bytes(&n.to_le_bytes()),
                        KernelArg::Bytes(&k.to_le_bytes()),
                        KernelArg::Bytes(&t.to_le_bytes()),
                        KernelArg::Buffer(packed_ptr),
                        KernelArg::Buffer(x_ptr),
                        KernelArg::Buffer(y_ptr),
                    ],
                )
                .expect("launch");
        }
        backend.synchronize(stream).expect("sync");
        start.elapsed().as_secs_f64() / iters as f64
    };
    launch(2); // warm
    let per_call = launch(8);
    let gmac = n as f64 * k as f64 * t as f64 / 1e9;
    eprintln!(
        "q1_0_gemm [{n}x{k}] T={t}: {:.2} ms/call — {:.2} T-MAC/s ({:.2} G-MAC)",
        per_call * 1e3,
        gmac / per_call / 1e3,
        gmac
    );
}

/// The f16-state GDN variants must agree with each other the same way
/// the f32 pair does (prefill == token-by-token decode walk), starting
/// from an identical half-precision state image.
#[test]
fn metal_gdn_prefill_f16_matches_decode_f16_walk() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let Ok(prefill) = backend.kernel("gated_delta_rule_prefill", "gated_delta_rule_prefill_f16")
    else {
        eprintln!("skipping: gated_delta_rule_prefill_f16 not in this kernel build");
        return;
    };
    let decode = backend
        .kernel("gated_delta_rule_decode", "gated_delta_rule_decode_f16")
        .expect("decode f16 kernel");

    let dim = 128usize;
    let num_k_heads = 2u32;
    let num_v_heads = 6u32;
    let t = 5usize;
    let qkv_stride = (2 * num_k_heads as usize + num_v_heads as usize) * dim;

    let qkv: Vec<half::bf16> = (0..t * qkv_stride)
        .map(|i| half::bf16::from_f32(test_val(i) * 0.5))
        .collect();
    let gate: Vec<f32> = (0..t * num_v_heads as usize)
        .map(|i| 0.55 + 0.4 * ((i % 7) as f32 / 7.0))
        .collect();
    let beta: Vec<f32> = (0..t * num_v_heads as usize)
        .map(|i| 0.3 + 0.6 * ((i % 5) as f32 / 5.0))
        .collect();
    let state0: Vec<half::f16> = (0..num_v_heads as usize * dim * dim)
        .map(|i| half::f16::from_f32(test_val(i * 7) * 0.02))
        .collect();
    let state0_bytes: Vec<u8> = state0.iter().flat_map(|h| h.to_le_bytes()).collect();

    let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };

    let qkv_ptr = backend.alloc(qkv.len() * 2).expect("alloc qkv");
    backend
        .copy_h2d(&bf16_slice_to_bytes(&qkv), qkv_ptr)
        .expect("h2d qkv");
    let gate_ptr = backend.alloc(gate.len() * 4).expect("alloc gate");
    backend.copy_h2d(&f32_bytes(&gate), gate_ptr).expect("h2d");
    let beta_ptr = backend.alloc(beta.len() * 4).expect("alloc beta");
    backend.copy_h2d(&f32_bytes(&beta), beta_ptr).expect("h2d");

    let state_ref = backend.alloc(state0_bytes.len()).expect("alloc state");
    let state_new = backend.alloc(state0_bytes.len()).expect("alloc state");
    backend.copy_h2d(&state0_bytes, state_ref).expect("h2d");
    backend.copy_h2d(&state0_bytes, state_new).expect("h2d");

    let z_dim = num_v_heads as usize * dim;
    let y_ref = backend.alloc(t * z_dim * 2).expect("alloc y_ref");
    let y_new = backend.alloc(t * z_dim * 2).expect("alloc y_new");

    let stream = backend.default_stream();
    let batch_one = 1u32;
    let dim_u32 = dim as u32;
    for tok in 0..t {
        let row = qkv_ptr.offset(tok * qkv_stride * 2);
        let k_off = num_k_heads as usize * dim * 2;
        let v_off = 2 * num_k_heads as usize * dim * 2;
        backend
            .launch_typed(
                decode,
                [num_v_heads, 1, 1],
                [128, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Buffer(state_ref),
                    KernelArg::Buffer(row),
                    KernelArg::Buffer(row.offset(k_off)),
                    KernelArg::Buffer(row.offset(v_off)),
                    KernelArg::Buffer(gate_ptr.offset(tok * num_v_heads as usize * 4)),
                    KernelArg::Buffer(beta_ptr.offset(tok * num_v_heads as usize * 4)),
                    KernelArg::Buffer(y_ref.offset(tok * z_dim * 2)),
                    KernelArg::Bytes(&batch_one.to_le_bytes()),
                    KernelArg::Bytes(&num_k_heads.to_le_bytes()),
                    KernelArg::Bytes(&num_v_heads.to_le_bytes()),
                    KernelArg::Bytes(&dim_u32.to_le_bytes()),
                    KernelArg::Bytes(&dim_u32.to_le_bytes()),
                ],
            )
            .expect("launch decode f16");
    }
    let t_u32 = t as u32;
    let stride_u32 = qkv_stride as u32;
    backend
        .launch_typed(
            prefill,
            [num_v_heads, 1, 1],
            [128, 1, 1],
            0,
            stream,
            &[
                KernelArg::Buffer(state_new),
                KernelArg::Buffer(qkv_ptr),
                KernelArg::Buffer(gate_ptr),
                KernelArg::Buffer(beta_ptr),
                KernelArg::Buffer(y_new),
                KernelArg::Bytes(&t_u32.to_le_bytes()),
                KernelArg::Bytes(&num_k_heads.to_le_bytes()),
                KernelArg::Bytes(&num_v_heads.to_le_bytes()),
                KernelArg::Bytes(&stride_u32.to_le_bytes()),
            ],
        )
        .expect("launch prefill f16");
    backend.synchronize(stream).expect("synchronize");

    let mut yr = vec![0u8; t * z_dim * 2];
    let mut yn = vec![0u8; t * z_dim * 2];
    backend.copy_d2h(y_ref, &mut yr).expect("d2h");
    backend.copy_d2h(y_new, &mut yn).expect("d2h");
    let yr = bytes_to_bf16_vec(&yr);
    let yn = bytes_to_bf16_vec(&yn);
    for i in 0..t * z_dim {
        let e = yr[i].to_f32();
        let a = yn[i].to_f32();
        // f16 state rounding differs between the register-resident
        // prefill walk and the store/reload decode walk — wider band
        // than the f32 pair.
        let tol = (e.abs() * 3e-2).max(1e-2);
        assert!(
            (e - a).abs() <= tol,
            "y[{i}] (tok {}): decode-f16 walk {e}, prefill-f16 {a}",
            i / z_dim
        );
    }
}
