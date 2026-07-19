<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Apple Metal Backend

Atlas can build and run on Apple Silicon (M1/M2/M3/M4) under the
`metal` cargo feature. The build links zero CUDA / NCCL and uses
`objc2-metal` bindings against the system Metal framework.

## Quick start

```sh
# Default (Linux/CUDA): unchanged.
cargo build -p spark-server --bin spark

# Apple Silicon: opt out of cuda, opt into metal.
cargo build -p spark-server --bin spark --features metal --no-default-features

# Sanity check — the binary should link no libcuda / libnccl.
otool -L target/debug/spark | grep -i cuda    # → no output
otool -L target/debug/spark | grep -i nccl    # → no output
```

## What's wired up

- **Build pipeline** — `kernels/metal/HARDWARE.toml` (`vendor =
  "apple"`) drives the existing `ComputeTarget` abstraction
  through `xcrun -sdk macosx metal -c → xcrun metallib`. Compiled
  metallib bytes are embedded into the runtime via
  `include_bytes!()`. Set `ATLAS_TARGET_HW=metal
  ATLAS_TARGET_MODEL=qwen3-5-4b-vlm-mlx-int8 ATLAS_TARGET_QUANT=mlx_int8`
  to compile the Qwen3.5-4B kernel set; macOS builds without
  these env vars auto-skip the kernel build (empty registry stub)
  so `cargo check` doesn't require a model directory.
- **Runtime backend** — `spark_runtime::metal_backend::MetalGpuBackend`
  implements every method of `GpuBackend` against `objc2-metal`:
  alloc/free via `MTLBuffer` (Shared storage on UMA), copy_h2d/d2h
  as memcpy against `buffer.contents()`, copy_d2d via
  `MTLBlitCommandEncoder`, kernel launch via
  `MTLComputeCommandEncoder.dispatchThreadgroups:`, streams as
  slabs of `(MTLCommandQueue, in-flight MTLCommandBuffer)`,
  events as `MTLSharedEvent`.
- **Pointer model** — `DevicePtr` carries the `MTLBuffer.gpuAddress()`
  u64 directly (Metal 3+ feature, native to all Apple Silicon),
  with a side `BTreeMap` for buffer lookup. Pointer arithmetic
  (`DevicePtr::offset(bytes)`) works as on CUDA — kernels see
  contiguous addressable memory.
- **Typed launch** — `GpuBackend::launch_typed(&[KernelArg])` is
  the canonical metal launch path. `KernelArg::Buffer(p)` maps to
  `setBuffer:offset:atIndex:`; `KernelArg::Bytes(b)` maps to
  `setBytes:length:atIndex:`. CUDA's untyped `launch()` keeps the
  default trait impl that flattens to `void**`.
- **MLX 8-bit weight format** — `spark_runtime::weights::mlx_int8`
  reads the `(.weight, .scales, .biases)` triplet that
  `mlx-community/<name>-MLX-8bit` exports use. Uint32-packed
  weights, BF16 scales/biases per group of 64. Provides
  `MlxInt8Weight::dequantize_to / gemv / gemm` wrappers around the
  fused-dequant kernels.

## Kernel inventory (`kernels/metal/common/`)

LLM trunk (Qwen3.5 full_attention layers):
```
embed_lookup        rms_norm            rope_apply
mlx_int8_dequant    mlx_int8_gemv       mlx_int8_gemm
kv_cache_append     attention_decode    attention_prefill
silu_gate           sigmoid_gate        bf16_add
argmax_bf16         softmax_topp
```

Linear-attention / SSM:
```
causal_conv1d_decode    selective_scan_decode
```

Vision tower (ViT-style):
```
layer_norm          dense_gemv_bf16     dense_gemm_bf16
attention_full      gelu                conv3d_patch_embed
```

TurboQuant KV cache (WHT-rotated, contiguous cache):
```
wht_bf16                  kv_cache_append_turbo8    attention_decode_turbo8
kv_cache_append_turbo4    attention_decode_turbo4
kv_cache_append_turbo3    attention_decode_turbo3
kv_cache_append_turbo2    attention_decode_turbo2
```

Every kernel has an FP32 CPU-reference parity test
(`metal_<name>_matches_reference`) within ≤2 BF16 ULPs.

## TurboQuant KV cache

The Metal contiguous cache supports the TurboQuant dtypes from the
CUDA backend's `--kv-cache-dtype` family, selected per `LayerKvCache`
via `MetalKvDtype` (the `metal_qwen35_inference` example exposes it as
`ATLAS_KV_DTYPE={bf16,turbo8,turbo4,turbo3,turbo2}`):

| dtype  | storage                                   | vs bf16 |
|--------|-------------------------------------------|---------|
| turbo8 | FP8 E4M3 + bf16 group-of-16 scales        | 2.13×   |
| turbo4 | 4-bit Lloyd-Max + FP8 scales (matched-L2) | 3.56×   |
| turbo3 | 3-bit Lloyd-Max (8 vals → 3 B) + FP8      | 4.57×   |
| turbo2 | 2-bit Lloyd-Max + FP8 scales              | 6.4×    |

Mechanics, mirroring the CUDA write path and decode bookends:

- `wht_bf16.metal` applies the canonical two-sided Rademacher rotation
  S2·H·S1 per head (`-DTQ_PLUS_SIGNS`, seed-42 tables byte-identical
  to `tq_plus_signs.cuh`; head_dim 128/256). The cache stores rotated
  values; the forward rotates Q before turbo attention and applies the
  inverse to the output.
- Quantizing appends use per-16-element groups; turbo4/3/2 store the
  matched-norm L2 scale (`||original|| / ||centroid_vec||`) in FP8.
- Decode attention dequantizes inline and applies the **sparse-V
  gate**: positions with unnormalized softmax weight ≤ 1e-3 skip the
  V dequant + accumulation entirely (attention-gated value
  dequantization — the benefit grows with context length).
- `Qwen35Kernels::resolve` hard-requires all turbo kernel handles, so
  a turbo cache can never silently fall back to the bf16 kernels.
- Per-quant `KERNEL.toml` `[build]` flags and `[modules]` entries are
  MERGED onto the common KERNEL.toml's (common first, model additions
  appended/winning per key) — model targets inherit `-ffast-math` and
  `-DTQ_PLUS_SIGNS` automatically.

Quality eval: `ATLAS_LOGITS_OUT=path` dumps per-step bf16 logits;
`tests/metal_kv_kld_compare.py` reports KLD + top-1 agreement between
two runs.

## Real-model integration tests

Four `#[ignore]`-gated tests exercise the kernels on actual
`mlx-community/Qwen3.5-4B-MLX-8bit` weights. Run with:

```sh
cargo test -p spark-runtime --no-default-features --features metal \
           metal_backend -- --include-ignored
```

The tests skip gracefully if the model isn't at
`~/models/Qwen3.5-4B-MLX-8bit` (override via
`$ATLAS_MLX_MODEL_DIR`):

- `metal_mlx_int8_dequant_real_model` — embed_tokens triplet.
- `metal_mlx_int8_gemv_real_model_q_proj` — layer-3 q_proj
  (8192 × 2560).
- `metal_real_model_chain_norm_then_qproj` — rms_norm → q_proj
  pipeline.
- **`metal_real_model_full_attention_block_layer3`** — the
  capstone: full LLM attention block (norm → QKV → per-head norm
  → RoPE → KV cache → attention → output gate → o_proj →
  residual → norm → SwiGLU FFN → residual) on actual layer-3
  weights.
- `metal_real_model_vision_block_forward` — full ViT block
  (norm1 → QKV+bias → attention_full → proj+bias → residual →
  norm2 → fc1+bias → gelu → fc2+bias → residual) on actual
  `vision_tower.blocks.0` weights.

## CI

`.github/workflows/ci.yml::test-macos-metal` runs on `macos-14`
on every PR:

1. `cargo check -p spark-server --features metal --no-default-features`.
2. `cargo test -p spark-runtime --features metal metal_backend`
   (every default-passing parity test).
3. `otool -L target/debug/spark` must list **no** `libcuda` or
   `libnccl` — guards against a stray `rustc-link-lib=cuda`
   slipping through any build.rs.

## Production serving: Bonsai-27B (Q1_0 GGUF)

The `spark serve` path now serves the Qwen3.5/3.6 GDN-hybrid dense
family end-to-end on Apple Silicon through `MetalGgufModel`
(`crates/spark-model/src/model/metal_gguf/`), which routes the
scheduler's `Model` trait through the vendor-agnostic
`forward::qwen3_5` per-layer orchestration. Validated with PrismML
**Bonsai-27B** (1-bit Q1_0 GGUF, 64 layers = 48 GDN + 16
full-attention, vocab 248320) on an M4/16 GB — 3.6 GB resident
weights, coherent chat, XML tool-calls (`qwen3_coder` parser +
constrained decoding), all three API surfaces, 2-seq concurrency:

```sh
# Build with the Bonsai kernel target embedded:
ATLAS_TARGET_HW=metal ATLAS_TARGET_MODEL=bonsai-27b ATLAS_TARGET_QUANT=q1_0 \
  cargo build --release -p spark-server --no-default-features --features metal

# Model dir: MLX-pack config.json (both `quantization` blocks stripped) +
# tokenizer files + Bonsai-27B-Q1_0.gguf (+ optional mmproj sidecar).
./target/release/spark serve --model-from-path ~/models/bonsai-27b-atlas \
  --port 8899 --max-seq-len 4096 --max-num-seqs 2 --max-batch-size 2 \
  --kv-cache-dtype bf16 --disable-thinking

# Smoke suite (9 tests: chat, coherence, tool calls, /v1/messages,
# count_tokens, /v1/responses, concurrency):
./scripts/test-bonsai-metal.sh
```

Key mechanics:
- **Q1_0 keep-packed** (`ATLAS_GGUF_NATIVE_Q1`, default-on for metal
  builds): projections, GDN in-projections (packed row-permute),
  out_proj (packed column-block permute — value_head_dim 128 = one
  block), embedding + LM head all stay 1-bit resident. Embed lookup
  is a CPU row dequant (UMA); LM head runs `q1_0_gemv` over the
  packed table.
- **Norm convention**: the loader normalizes GGUF norms to the
  zero-centered HF form for CUDA's `x·(1+w)/rms` kernels; the metal
  `rms_norm` is vanilla `x·w/rms`, so `MetalGgufModel` re-adds the +1
  at init (`norm_plus_one`). GDN `linear_attn.norm` ships vanilla and
  stays raw.
- **Kernels**: `q1_0_gemv(+_batchm)`, fused `q1_0_gemv_gate_up` and
  `q1_0_gemv_silu_gate(_resid)` (`kernels/metal/common/`), parity-
  tested against the loader's CPU oracle (`parity_q1.rs`).
- Per-token prefill / serial decode (weight-bandwidth-bound); batched
  prefill via `attention_prefill` + GEMM is the open perf follow-up.

## What's still required beyond text

| Lift | Notes |
|---|---|
| Vision (mmproj) forward | The Q8_0 clip sidecar already loads into the store (`model.visual.*`, BF16) and the ViT kernels are parity-tested; `MetalGgufModel::prepare_vision_embed` + the prefill embedding splice + MRoPE image positions are the gap |
| Batched prefill | Per-token prefill runs ~6 tok/s at 27B; the llama.cpp Metal reference does ~130 tok/s with real GEMM prefill |
| Token-level parity harness vs `llama-cli` reference | Greedy-decode comparison against the PrismML llama.cpp fork on the same GGUF |
EOF