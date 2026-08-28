# qwen4_exp (Qwen3.8-Flash-Next) port tools

Checkpoint-inspection tools for the port tracked in
[Avarok #753](https://github.com/Avarok-Cybersecurity/atlas/issues/753).
Both read `RadixArk/Qwen3.8-Flash-Next-NVFP4` off disk; set `SNAP` at the top
of each if your snapshot path differs.

## `ns_audit.py` — the loader's checklist

Collapses the checkpoint's 296,475 tensors into ~111 name families and
assigns each a destination keyed to the issue's work items. **A loader that
silently skips a family produces a model that runs and is wrong**, so the
useful artifact is the complete list with an explicit destination for every
family — including the ones being dropped on purpose (MTP).

Current state: all 111 families classified, none unclassified.

```
 families   tensors  destination (issue #753 item)
        2         2  A  embed/head/norm
       11       387  B  mHC
       11       138  C  PLE n-gram
        3        36  D  QSA indexer
        6        72  E  attention
        9       324  F  GDN
       12    294912  G  MoE experts
        5       240  G  MoE shared/router
       21       333  H  vision (have)
       31        31  I  MTP (drop v1)
      111    296475  TOTAL
```

Two architectural facts this surfaced that the config alone does not:

- **There are no standalone `input_layernorm` / `post_attention_layernorm`
  tensors, and no final `model.norm`.** Normalization lives inside the
  hyper-connection blocks (`hc_norm`), and the model-level
  `hyper_connection_mixer` — which collapses the 4 residual streams back to
  one before `lm_head` — carries the final norm. A loader looking for the
  usual per-layer norms will find nothing and must not paper over it.
- **The PLE block has a `conv1d`**, matching `ple_conv_kernel_size: 4`. It is
  not a plain embedding lookup.

## `ckpt_split.py` — does it fit on one Spark?

Sums real file sizes per family (local blobs, Hub `Content-Length` for
anything still downloading) and prints the resident footprint with the PLE
n-gram tables deferred to the NVMe row cache.

```
    63.32 GB   192  routed experts (NVFP4)
    47.68 GB    10  PLE n-gram tables (FP8)
    14.91 GB     4  backbone (BF16)
   125.91 GB   206  TOTAL
resident if PLE deferred to NVMe: 78.23 GB   (budget: 97.3 GB at 0.80 util)
```

Note `index.json`'s `total_size` is **bytes** (135,195,303,851 = 125.91 GiB);
reading it as GB is how you get a phantom 9 GB discrepancy.

## Serving the RadixArk release on one GB10 — the three snapshot builders

The dev-box recipe served the Inferact release (BF16 PLE, one file). The
RadixArk release needs a one-time preprocessing pass, because its PLE table
ships FP8 across 10 files (the row cache wants BF16 in one file) and its MTP
block ships fused BF16 (the loader wants per-expert ModelOpt NVFP4). Run, in
order (any Python with numpy; `QWEN4EXP_SRC_SNAPSHOT` overrides the HF-cache
glob):

1. `convert_ple_bf16.py <out.safetensors>` — dequantizes the 128 FP8 n-gram
   shards with the checkpoint's scalar `weight_scale` into ONE BF16
   safetensors file (102.4 GB, ~5 min), the exact layout
   `weight_loader/qwen4_exp/ple.rs` + the segmented NVMe row cache expect.
   Spot-checks its own output.
2. `make_mtp_extra.py` — slices the fused MTP experts
   (`gate_up_proj [512,1280,2560]`, `down_proj [512,2560,640]`) per expert
   and quantizes to ModelOpt NVFP4 (1.6 GB), replicating
   `quantize_bf16_to_nvfp4.cu`'s scale + E2M1 rounding math bit-for-bit;
   every other `mtp.*` tensor passes through BF16. Output name matters:
   `extra_weights.safetensors` rides the loader's extra-weights hook, which
   bypasses the main shards' `skip_mtp`.
3. `build_serving_snapshot.py` — assembles a serving snapshot: symlinks the
   original files minus the 10 `model-plefp8-*` shards, adds the two files
   above, patches `model.safetensors.index.json` to point the PLE shard
   entries at the BF16 file (the scalar-scale entry is dropped — the BF16
   path never reads it).

Serve fingerprint that measured 45.8 tok/s code / 39.4 prose / 53.1 counting
(gx10-ecdf, 2026-08-28, commit bdfc322b, n=3 steady-state streamed timing,
thinking=low, temp 0 — preliminary, single-harness):

```
ATLAS_QWEN4EXP_BF16_GDN=0 ATLAS_GDN_SLIM=1 ATLAS_CUDA_HEADROOM_MB=3072 \
ATLAS_DFLASH_SPEC_THINK=1 ATLAS_PLE_MAX_TOKENS=2176 \
spark serve --model-from-path <snapshot> --kernel-target qwen3.8-flash-next \
  --max-seq-len 8192 --max-num-seqs 2 --max-batch-size 2 \
  --gpu-memory-utilization 0.84 --kv-cache-dtype bf16 \
  --speculative --num-drafts 2 --ssm-cache-slots 2 \
  --max-prefill-tokens 2048 --enable-prefix-caching
```

Memory campaign (2026-08-28, commits 01aec27e/4fd3fe77): slab-packed weight
upload (the driver's 2 MiB granule padding on ~150k per-expert tensors was
~10 GB), dead-weight retirement (GDN BF16 originals + BF16 lm_head, 5.05 GB,
poison-validated), and `ATLAS_NO_VISION=1` (~1 GB) take pre-KV from 93.1 to
~76 GB — the KV pool at util 0.84 is then ~21.5 GB ≈ 940k tokens even at
`--max-seq-len 200000`. Long-context serving needs two knobs the 8k profile
didn't: `ATLAS_QSA_MAX_TOKENS` ≥ max-seq-len (QSA per-seq key buffers,
~58 MB/attention-layer/seq at 200k, lazily allocated) and a `--request-timeout`
sized for chunked-prefill TTFT (a 119k-token prompt prefills in ~22 min on
gx10-ecdf). Decode after the campaign: 47.8 tok/s median code, 41.6 prose
(n=3 steady-state streamed, within the 47–50 boot-to-boot band).

K sweep on the code prompt: K=2 41.2, **K=3 45.8**, K=4 37.1 (verify cost
outruns depth). The batched-verify-attention commit (ms body,
QSA ingest batched through prefill_ingest) lifts K=3 code to **49.4**.
Graph pricing (ATLAS_NO_DECODE_GRAPHS A/B): graphs are NEUTRAL at C=1 and
only recover a C>1 eager-shape penalty — the serial floor is intra-kernel
latency, and the routed-expert reads that scale with K are the verify's one
non-amortizable bandwidth term. `ATLAS_DFLASH_SPEC_THINK=1` is the whole payoff on this
always-thinking model and inherits the known batch-K T=0 numerics floor;
the agentic-gate measurement for spec-in-think on THIS model is still owed.
