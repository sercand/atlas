# Atlas memory-reduction campaign — GB10, 2026-08

**Goal.** Reduce pre-KV GPU footprint on `unsloth/Qwen3.8-27B-NVFP4` from ~48.7 GiB
toward ~30 GiB without regressing decode throughput.

Work items in §6 are taken strictly in order, one at a time. After every item:
re-measure per §4, append a row to §7, commit only if the item both saved memory
and held throughput. An item that regresses throughput is reverted, with the
measurement written down.

---

## 1. Environment

- Repo `/home/otsimo/work/atlas-ir`, branch `test/ladder-stack`.
- Host **`gx10-ecdf`** (GB10, 121.6 GB unified memory, sm_121). Verify with `hostname`.
- **`BENCH.toml`'s committed gates were calibrated on `dgx1`, a different box.**
  Absolute gate floors do not transfer. Only same-box before/after A/B is valid.
  Commit `9ce07fea` retracts a regression that turned out to be dgx1-vs-dgx2
  confusion; do not repeat it.
- **A vLLM co-tenant shares this GPU and is not stable.** Observed within one
  hour on 2026-08-17: 28.8 → 33.1 → 33.9 GiB, then a restart down to 21.3 GiB.
  Anything measured by differencing free memory inherits that movement — see §3.

Every cargo and serve invocation needs the NCCL shim on both paths, **including
the `spark benchmark` client** (it exits with a bare `libnccl.so.2` loader error
otherwise, which is easy to mistake for an empty result):

```bash
export LIBRARY_PATH=$HOME/nccl-shim LD_LIBRARY_PATH=$HOME/nccl-shim
```

Type-check (~30 s; `cuda` is a default feature, so this does check CUDA paths):

```bash
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 \
LIBRARY_PATH=$HOME/nccl-shim LD_LIBRARY_PATH=$HOME/nccl-shim \
cargo check -p spark-model --message-format short
```

Release build (~35 s incremental; `ATLAS_SKIP_BUILD` would produce a binary with
zero PTX modules that cannot serve):

```bash
ATLAS_TARGET_MODEL=qwen3.8-27b CUDARC_CUDA_VERSION=13000 \
LIBRARY_PATH=$HOME/nccl-shim LD_LIBRARY_PATH=$HOME/nccl-shim \
cargo build --release -p spark-server --bin spark
```

**`/usr/local/bin/spark` is root-owned, so the build canNOT hardlink into it.**
The fresh binary is `target/release/spark`; `/usr/local/bin/spark` silently stays
whatever was installed last. Always confirm which binary you are measuring:

```bash
ls -la target/release/spark /usr/local/bin/spark
strings <binary> | grep -c "mem-report"   # non-zero => has the §2 instrumentation
```

---

## 2. Landed work

### `cc165115` — three load-time leak fixes + `ATLAS_ALLOC_HISTO`

Cut 8.28 GiB and 160 allocations. See the commit message for the full argument.

1. **1.99 GiB** — `qwen35_dense.rs` `CompressedTensors` `load_nvfp4` dequantised
   FP8→BF16, requantised to NVFP4, and never freed the BF16. The sibling
   `Standard | Fp8Dequanted` arm always did, citing Atlas issue #A1; this arm —
   the one mixed-precision checkpoints take — was missed.
2. **1.4 GiB** — `set_fp8_prefill_only_weights` overwrote the `out_proj_fp8`
   pointer `predequant_for_prefill` had just allocated, without freeing it.
   30 MiB × 48 SSM layers, every load, no flag needed.
3. **3.75 GiB** — the QKVZ FP8 copy is dead under `ATLAS_FP8_ROWWISE`; skipped.

### Item 6 (this campaign's first item) — see §7

---

## 3. THE INSTRUMENT — read before quoting any number

### 3.1 Memory: use the allocation ledger, NOT the free-memory delta

The `Atlas-own` line in the serve log is computed as
`free-at-context-init − free-now`. On a box with a moving co-tenant that is a
**noisy instrument**, and the campaign's original 43.4 GiB baseline came from it.

Controlled demonstration, 2026-08-17, same host, same serve profile:

| binary | `Atlas-own` (free-delta) | ledger `pre-KV footprint` |
|---|---|---|
| baseline (`cc165115`) | 50.0 GiB | *(not implemented)* |
| item 6 | 48.5 GiB | **48.71 GiB** |
| item 6, repeat | 52.1 GiB | **48.71 GiB** |

Same binary, same profile, two runs: the free-delta moved **3.6 GiB**; the ledger
was **bit-identical**. The ledger sums each allocation's requested size and is
co-tenant-immune by construction. In the one run where the co-tenant happened to
hold still, the two agreed to 0.4% (48.71 vs 48.5), which is the cross-check that
the ledger is measuring the right thing.

**The recorded 43.4 GiB baseline does not reproduce** — the same baseline binary
reads 50.0 GiB today. Treat 43.4 as unfingerprintable and superseded. The
campaign's memory reference is now **48.71 GiB (ledger)**.

Read it from the serve log without any flag:

```
INFO ... mem_report: pre-KV footprint: 48.71 GiB resident, of which 47.07 GiB is weights (checkpoint on disk: 21.81 GiB)
```

### 3.2 Throughput: `decode-floor`, and it is stable

Fingerprint: host `gx10-ecdf` · checkpoint `unsloth/Qwen3.8-27B-NVFP4` rev
`16b6615af3548b88e2d8e382457bc705b00479cf` · serve profile below verbatim ·
2026-08-17.

```bash
env LIBRARY_PATH=$HOME/nccl-shim LD_LIBRARY_PATH=$HOME/nccl-shim \
ATLAS_MTP_ACCEPT_DEBUG=1 \
ATLAS_PREFILL_CODISPATCH=1 ATLAS_FP8_ROWWISE=1 ATLAS_MTP_DCUT_RATIO=1.0 \
ATLAS_MTP_K_LADDER=1:3,2:1,4:2,8:2,16:1 \
<binary> serve unsloth/Qwen3.8-27B-NVFP4 \
  --host 127.0.0.1 --port 36200 --model-name Qwen3.8-27B \
  --max-seq-len 2048 --max-batch-size 2 --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 --enable-prefix-caching true \
  --ssm-cache-slots 8 --ssm-checkpoint-interval 32 \
  --vision-max-pixels 16384 \
  --speculative --num-drafts 3 --mtp-quantization bf16 \
  --scheduling-policy fifo --tool-call-parser qwen3_coder --disable-tool-grammar true \
  --request-timeout 0 --vision-allow-remote-images \
  --ssm-h-dtype f16-pool --gdn-fused-norm --ssm-batched-recurrent \
  --ssm-tail-midchunk false --mtp-gate force --prefill-varlen-batch --no-tui
```

Kept verbatim as `serve-profile.sh` beside this file — take the profile from
there rather than retyping it, so the one-variable rule is mechanical:

```bash
docs/campaigns/gb10-memory-2026-08/serve-profile.sh target/release/spark [extra flags]
```

**Never measure throughput with `--mem-report` set**: it captures a backtrace per
allocation over 4 MiB for the life of the process, not just during load.

| Metric | Reference | Notes |
|---|---|---|
| **decode-floor median server tok/s** | **31.1** | 31.2/31.1/31.0, baseline binary, 2026-08-17 |
| decode-floor output tokens | 604 every run | Deterministic. A different value means the instrument moved. |
| decode-floor accepted drafts | 400 every run | `tok_step = 604/204 = 2.96`, near the k=3 ceiling. Proves MTP engaged. |

Regression rule: median must stay **≥ 30.8**, AND out-tok must be 604 AND
accepted must be 400. If either of the latter two moves, the instrument changed
and the tok/s is not comparable.

---

## 4. How to measure (after every item)

```bash
# 1. build (§1) — confirm you are running target/release/spark, not /usr/local/bin
# 2. start the §3.2 profile, wait for /v1/models
# 3. memory: read the "pre-KV footprint" line from the serve log (no flag needed)
# 4. throughput — 59 s, WITHOUT --mem-report:
export LIBRARY_PATH=$HOME/nccl-shim LD_LIBRARY_PATH=$HOME/nccl-shim
spark benchmark run decode-floor --url http://127.0.0.1:36200 --model Qwen3.8-27B --yes
# 5. attribution, only when deciding what to cut next — separate run, WITH --mem-report
```

Records land in `~/.atlas/runs/<benchmark>/run-*.json` — cite those IDs, not prose.

**Expect `decode-floor` to report `INCONCLUSIVE`.** That is correct, not a
failure: it requires ≥750 output tokens and this box's natural stop is 604. The
floor was calibrated to dgx1's 915-token stop. Read the median tok/s out of the
table and ignore the verdict. Do **not** "fix" this by changing the fixture —
that would destroy comparability.

Before writing any perf number anywhere, invoke the `measurement-discipline`
skill. It is mandatory in this repo and encodes a real incident.

---

## 5. Traps

1. **`--gpu-memory-utilization` is a fraction of TOTAL memory (121.6 GB)**, and
   `kv_budget` is additionally clamped by `.min(actual_free − inference_reserve)`
   (`crates/spark-model/src/factory/build.rs`). On a box with co-tenants, raising
   util does nothing. At util 0.40 this model fails to start outright.
2. **`--vision-max-pixels 16384` is an AREA** (128×128 px → 64 patches), not a
   patch count.
3. **The 4.74 GiB `2 × 2425 MiB` bucket is NOT ViT scratch.** Item 6's attribution
   shows `vision_encoder` holding 0.00 GiB across 16 allocations under the §3.2
   profile, while one of the two 2425 MiB buffers is
   `weights.lm_head / quant_helpers::dequant_fp8_blockscaled_to_bf16`. A prior
   note claiming the §3.2 profile "already avoids" this 4.74 GiB was wrong on
   both counts: it is present, and it is lm_head, not vision. See §6 item 1.
4. **A human-quoted table claimed C=1 23.59 / C=2 38.95 / C=4 74.21.** C=1 and
   C=2 reproduce (22.5 / 39.1). **C=4 does not and cannot**: the §3.2 profile sets
   `--max-batch-size 2`, so at C=4 two requests queue, TTFT p50 hits 17.6 s, and
   aggregate flatlines at the C=2 value. That row came from a different profile.
5. **This checkpoint is not mostly NVFP4** despite its name. 21.81 GiB on disk:
   only ~7.0 GiB is 4-bit (MLP gate/up/down L0–55); ~10.8 GiB is FP8 E4M3
   per-channel (attn q/k/v/o, SSM `in_proj_qkv`/`in_proj_z`/`out_proj`, `lm_head`,
   MLP L56–63); ~3.3 GiB BF16. `config.json` has two `config_groups`. The FP8
   tensors trigger the dequant→requant path that manufactures derived copies.
6. **`WeightStore` is released only at model teardown**
   (`crates/spark-model/src/model/types.rs`), never as layers consume it. Correct
   for genuinely aliased tensors (norms, `conv1d`, and the `ATLAS_FP8_ROWWISE`
   weights, which read store bytes directly for the model's lifetime) — which is
   why item 3 needs a real ownership split and not a blanket free.
7. **Thinking gates speculation OFF** (`!inside_thinking`). Any measurement
   without `reasoning_effort:"none"` measures a blend of the serial floor and the
   engine. `decode-floor` sets it for you.
8. **Stop the server by PORT OWNER, not by process name — `stop.sh` beside this
   file does it.** This bit once, and the failure is silent in the worst way:
   `pkill -x spark` does not match the baseline binary copied to
   `spark-baseline-…`, so the old server kept the port, the new one died in
   preflight on "inference buffers alone need 6.96 GB but only 3.94 GB is free",
   and `decode-floor` cheerfully measured the OLD binary while being recorded as
   the new one. It was caught only because the two "different" builds produced
   suspiciously identical numbers and the serve log was re-read. Two habits make
   it non-silent: confirm the `pre-KV footprint` line in the NEW server's log
   before benchmarking against it, and wait for the GPU memory to actually come
   back — the listener closes well before teardown finishes sweeping ~1500
   allocations.

---

## 6. Work items

Measured composition at 48.71 GiB (item 6 attribution, §3.2 profile, ledger):

| GiB | allocs | owner |
|---:|---:|---|
| 21.69 | 662 | `weights.store / fast_weights::load_shard_fast` — the checkpoint itself, 1.0x |
| 8.96 | 192 | `weights.layers / dense_ffn::DenseFfnLayer::ensure_nvfp4_mmq_weight` |
| 4.64 | 224 | `weights.layers / weight_map::loaders_fp8::quantize_to_nvfp4` |
| 3.75 | 48 | `weights.layers / qwen35_dense::rowwise_fp8::concat_fp8_per_row` |
| 3.52 | 176 | `weights.layers / quantized::QuantizedWeight::transpose_for_gemm_gs` |
| 2.37 | 1 | `weights.lm_head / quant_helpers::dequant_fp8_blockscaled_to_bf16` |
| 1.41 | 48 | `weights.layers / ModelWeightLoader>::load_layers` |
| 0.97 | 18 | `buffer_arena / buffers::BufferArena::new` |
| 0.67 | 2 | `weight_map::loaders_fp8::quantize_to_nvfp4` (outside any scope) |
| 0.74 | 2096 | everything else, all under 0.62 GiB per owner |

### Item 6 — allocation labels + dump-at-end-of-load — **DONE, see §7**

### Item 3 — per-tensor `WeightStore` release — **DONE (−8.49 GiB)**

Landed: `WeightStore::retire(name, copies)` + `free_retired(gpu)`, and retirement
of 185 store tensors totalling 8.49 GiB:

| tensors | GiB | what | retired at |
|---:|---:|---|---|
| 96 | 3.75 | SSM `in_proj_qkv` + `in_proj_z` FP8 | `qwen35_dense.rs`, after both consumers copy |
| 64 | ~1.56 | attn q/k/v/o FP8 | the `CompressedTensors` `load_nvfp4` closure |
| 24 | ~1.99 | MLP L56–63 FP8 tail | `quantized_from_fp8` |
| 1 | 1.18 | `lm_head` FP8 | `loaders_b::load_lm_head` |

Two corrections to the original plan:

* **Trap 6 was over-broad.** It said the `ATLAS_FP8_ROWWISE` SSM
  `in_proj_qkv`/`_z`/`out_proj` FP8 bytes are all read for the model's lifetime.
  Only `out_proj` is: `load_fp8_per_row` aliases the store, but
  `concat_fp8_per_row` `copy_d2d`s `in_proj_qkv` + `in_proj_z` into one fresh
  `[Q|K|V|Z]` buffer and only that is kept. `out_proj` has no concat, so its
  alias stays live and it must NOT be retired.
* **The "unattributed 48 × 50 MiB = 2.34 GiB" bucket was `in_proj_qkv`.**
  `[10240, 5120]` FP8 = 50 MiB; `in_proj_z` is `[6144, 5120]` = 30 MiB; together
  80 MiB/layer, matching the concat output size exactly.

Deliberately NOT retired:

* **SSM `out_proj` FP8** — `load_fp8_per_row` aliases it and, unlike the
  `in_proj` pair, there is no concat to copy it away, so the alias stays live for
  prefill for the model's lifetime.
* **The 2.37 GiB `weights.lm_head / dequant_fp8_blockscaled_to_bf16` BF16
  buffer** — that is the live head the logits GEMM reads, not a leftover. It is
  a fair target for a *different* change (serve the head from NVFP4 and drop the
  BF16), not for retirement.
* **`quantized_any`'s `Bf16Raw` arm**, which frees a store pointer eagerly
  (`nvfp4_detect.rs`). Converting it to `retire` would be tidier and would end a
  latent double-free with `release`, but its eager free is load-bearing for PEAK
  memory: the comment records a 35B BF16 MoE hitting ~109 GB pre-KV without it,
  and deferred retirement would restore exactly that peak. Leave it.

**Method — use the poison probe, it is the point.** `ATLAS_POISON_RETIRED_WEIGHTS=1`
overwrites retired buffers with `0xA5` and does not free them. A use-after-free
otherwise reads bytes that merely happen to still be intact, so it passes every
test and corrupts later; a stale reader of poison is wrong immediately. Retire,
run the probe, check generation is coherent AND decode-floor still reports
604/400, and only then trust the free.

### Item 4 — `out_proj` copy set *(~2.8 GiB)*
192 × 30 MiB = 4 copies per SSM layer. Enumerate which are live — predequant FP8,
rowwise (store alias), NVFP4, NVFP4 `_t` — and reduce to two. `out_proj_fp8` IS
read by batched decode at
`crates/spark-model/src/layers/qwen3_ssm/trait_decode_batched.rs:1087`, so this
changes verify numerics: run decode-floor **and** check accepted-drafts stays 400.

### Item 2 — collapse the dual weight layout *(~9.5 GiB, HIGHEST PERF RISK)*
Now precisely located: 8.96 GiB in `ensure_nvfp4_mmq_weight` and 3.52 GiB in
`transpose_for_gemm_gs`. `kernels/gb10/common/w4a16_gemm.cu:16-19` documents that
`w4a16_gemm` and `w4a16_gemm_t` are the same math — the `_t` form exists for
"coalesced N-dim reads for better LPDDR5X bandwidth", so the duplication is a
bandwidth optimisation, not a functional requirement. Two options:
- (a) drop `_t`, dispatch the non-transposed kernel — simple, costs prefill bandwidth;
- (b) keep ONE arena-owned transpose scratch reused across layers (one layer's
  worth instead of 64) — pays a transpose per layer per prefill, which may vanish
  under compute-bound prefill at large M.
Prototype **behind a flag**, default off, and A/B both prefill (TTFT) and decode.

### Item 7 — util semantics + error message *(no perf risk)*
`factory/build.rs` bails with "Raise --gpu-memory-utilization", which provably
does not help when co-tenants hold the memory. Name the co-tenant bytes and
actual free instead.

### Item 1 — lm_head/ViT quadratic scratch *(4.74 GiB — RESCOPED, do LAST)*
Originally written as "lazy ViT scratch". Item 6 disproved the ViT attribution
(trap 3). The 4.74 GiB is `2 × 2425 MiB`, one of which is the `lm_head` FP8→BF16
dequant. Re-derive what the second one is from a `--mem-report` run before
planning work here.

### Item 8 — investigate unsloth-specific loading
Does the unsloth mixed FP8/NVFP4 layout carry metadata implying a load path Atlas
is missing that would avoid the dequant→requant derived copies entirely? Check
`weight_scale` shapes and whether a per-channel FP8 GEMM could serve decode
directly. Read the module doc of
`crates/spark-model/src/weight_loader/qwen35_dense/rowwise_fp8.rs` first —
cuBLASLt FP8 row-wise is dead on sm_121 (heuristic status 15).

---

## 7. Results log — append one row per item, do not edit prior rows

| Item | pre-KV GiB (ledger) | decode tok/s (median of 3) | out tok | accepted | run record | verdict |
|---|---|---|---|---|---|---|
| baseline (`cc165115`) | n/a — free-delta read 50.0 | 31.1 (31.2/31.1/31.0) | 604 | 400 | `run-1787001820494991508` | reference |
| 6 — labels + report + tripwire | 48.71 | 31.1 (30.8/31.2/31.1) | 604 | 400 | `run-1787001983979160734` | **KEEP** — no memory change expected or seen; throughput unchanged |
| 3a — retire SSM `in_proj` FP8 originals | 44.96 (−3.75) | 31.1 (31.1/31.0/31.1) | 604 | 400 | `run-1787002925279178569` | superseded by 3b |
| 3a — poison probe (control) | 48.71 (poisoned, not freed) | 31.1 (31.1/31.0/31.1) | 604 | 400 | `run-1787002817059208139` | safety gate passed |
| 3b — + attn q/k/v/o, MLP tail, `lm_head` | **40.22** (−8.49) | **31.1** (median of 6) | 604 | 400 | `run-1787004106534527817`, `run-1787004166429407764` | **KEEP** — 185 tensors, −8.49 GiB, ratio 2.16x → 1.77x |
| 3b — poison probe (control) | 48.71 (poisoned, not freed) | 31.3 (31.3/31.2/31.3) | 604 | 400 | `run-1787003363972615199` | safety gate passed |

Item 3b A/B, both on a co-tenant-free box, `decode-floor` server tok/s:

| | n | median | mean | sd |
|---|---:|---:|---:|---:|
| baseline (`cc165115`) | 12 | 31.10 | 30.98 | 0.27 |
| item 3b | 6 | 31.10 | 31.12 | 0.04 |

Identical medians; item 3b's mean is marginally higher and its spread smaller.
No regression. 604 output tokens and 400 accepted drafts on every single run.

Superseded: the pre-campaign `43.4 GiB / 31.3 tok/s` reference
(`run-1787000030318474839`). Throughput reproduces (31.1 vs 31.3, within σ≈0.15×2).
The 43.4 GiB memory figure does not — see §3.1 for the mechanism and the
controlled A/B that supersedes it.

---

## 8. A larger prize, outside the memory scope

A human-supplied log from this same engine and serve profile showed
`mean_na=1.410, serial=0.20, 415 tokens, finish=stop, 22.8 tok/s`. The §3.2
instrument on the same binary shows `mean_na=1.96, serial≈0, tok_step 2.96,
31.1 tok/s`. Same engine — the entire difference is `reasoning_effort:"none"` and
prompt class. Note `mean_na` is computed over MTP steps only
(`crates/spark-server/src/scheduler/mtp_accept_debug.rs:212`), so that run's true
blended rate was `0.80 × 2.410 + 0.20 × 1.0 = 2.13` tokens/step, not 2.41.

If real traffic looks like that log rather than like the fixture, raising draft
acceptance is worth more than every memory item combined. Its `K4 summary` showed
23 of 100 steps rejecting every draft at `k_drafts=3`. Treat as a separate
campaign; do not fold it into this one, and do not let it contaminate the
memory A/Bs.
