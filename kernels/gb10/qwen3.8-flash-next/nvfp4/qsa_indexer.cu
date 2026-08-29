// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.8-Flash-Next QSA indexer — the decode-side selection machinery.
//
// Reference: modeling_qwen4_exp.py Qwen4ExpTextQSAIndexer. Per query, the
// visible prefix is grouped into `ratio`(=4)-token blocks; each block's key
// is the MEAN of its raw per-token indexer keys, then k_layernorm
// (offset-from-1 RMSNorm), then partial rope at the block's FIRST token
// position. Scores are sum_h relu(q_h . k_b) / sqrt(head_dim); the top
// `block_topk` blocks plus the incomplete tail are the visible set.
//
// Selection feeds the EXISTING paged decode attention: qsa_gather packs the
// selected tokens' K/V rows into a contiguous scratch laid out NHD
// ([page, slot, kv_head, dim]) so an identity block table over the scratch
// reproduces the reference mask semantics with zero new attention code.
//
// Rope here is computed INLINE in double precision (32 freq lanes,
// inv_freq_j = theta^(-2j/rot)) rather than read from the attention rope
// tables — the golden's cos/sin come from torch fp32 and double sincos
// keeps the parity comparison out of ulp territory. Text-only mrope with
// equal position grids reduces to exactly this.

#include <cuda_bf16.h>

__device__ __forceinline__ float qsa_block_reduce_sum(float v, float* red) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    }
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float tot = 0.0f;
    if (threadIdx.x == 0) {
        const unsigned int warps = (blockDim.x + 31) >> 5;
        for (unsigned int w = 0; w < warps; ++w) tot += red[w];
        red[0] = tot;
    }
    __syncthreads();
    return red[0];
}

// normed (already in smem, length hd) -> rope at `pos` -> out (bf16).
// Assumes hd threads; rot must be even, pairs are (j, j + rot/2).
__device__ __forceinline__ void qsa_rope_store(
    const float* normed, __nv_bfloat16* out,
    unsigned int d, unsigned int rot, unsigned int pos, float theta
) {
    if (d < rot) {
        const unsigned int half = rot >> 1;
        const unsigned int j = (d < half) ? d : d - half;
        const double inv_freq = exp(-2.0 * (double)j / (double)rot * log((double)theta));
        double s, c;
        sincos((double)pos * inv_freq, &s, &c);
        const float x1 = normed[j];
        const float x2 = normed[j + half];
        const float v = (d < half) ? (x1 * (float)c - x2 * (float)s)
                                   : (x2 * (float)c + x1 * (float)s);
        out[d] = __float2bfloat16(v);
    } else {
        out[d] = __float2bfloat16(normed[d]);
    }
}

// ── qsa_block_pool ──
// Pool `n_new` freshly COMPLETE blocks starting at `first_block`:
// mean(ratio raw keys) -> RMSNorm*(1+w) -> rope at pos = block*ratio.
// Appends into block_keys [*, hd]. Grid: (n_new,1,1)  Block: (hd,1,1).
extern "C" __global__ void qsa_block_pool(
    const __nv_bfloat16* __restrict__ raw_keys,   // [S, hd]
    const __nv_bfloat16* __restrict__ k_norm_w,   // [hd]
    __nv_bfloat16* __restrict__ block_keys,       // [max_blocks, hd]
    const unsigned int first_block,
    const unsigned int ratio,
    const unsigned int hd,
    const unsigned int rot,
    const float theta,
    const float eps
) {
    const unsigned int b = first_block + blockIdx.x;
    const unsigned int d = threadIdx.x;

    extern __shared__ float smem[];               // [hd] normed + red
    float* stage = smem;
    float* red = smem + hd;

    float v = 0.0f;
    for (unsigned int r = 0; r < ratio; ++r) {
        v += (float)raw_keys[(size_t)(b * ratio + r) * hd + d];
    }
    v /= (float)ratio;

    const float sq = qsa_block_reduce_sum(v * v, red);
    const float rms = rsqrtf(sq / (float)hd + eps);
    stage[d] = v * rms * (1.0f + (float)k_norm_w[d]);
    __syncthreads();

    qsa_rope_store(stage, block_keys + (size_t)b * hd, d, rot, b * ratio, theta);
}

// ── qsa_qprep ──
// One decode query: per head, RMSNorm*(1+w) then rope at `pos`.
// q_in is the head-concatenated slice of the qk projection row.
// Grid: (n_heads,1,1)  Block: (hd,1,1). Output FP32 (feeds the scorer).
extern "C" __global__ void qsa_qprep(
    const __nv_bfloat16* __restrict__ q_in,       // [n_heads, hd]
    const __nv_bfloat16* __restrict__ q_norm_w,   // [hd]
    float* __restrict__ q_out,                    // [n_heads, hd]
    const unsigned int hd,
    const unsigned int rot,
    const unsigned int pos,
    const float theta,
    const float eps
) {
    const unsigned int h = blockIdx.x;
    const unsigned int d = threadIdx.x;

    extern __shared__ float smem[];
    float* stage = smem;
    float* red = smem + hd;

    const float x = (float)q_in[(size_t)h * hd + d];
    const float sq = qsa_block_reduce_sum(x * x, red);
    const float rms = rsqrtf(sq / (float)hd + eps);
    stage[d] = x * rms * (1.0f + (float)q_norm_w[d]);
    __syncthreads();

    float* out = q_out + (size_t)h * hd;
    if (d < rot) {
        const unsigned int half = rot >> 1;
        const unsigned int j = (d < half) ? d : d - half;
        const double inv_freq = exp(-2.0 * (double)j / (double)rot * log((double)theta));
        double s, c;
        sincos((double)pos * inv_freq, &s, &c);
        const float x1 = stage[j];
        const float x2 = stage[j + half];
        out[d] = (d < half) ? (x1 * (float)c - x2 * (float)s)
                            : (x2 * (float)c + x1 * (float)s);
    } else {
        out[d] = stage[d];
    }
}

// ── qsa_score ──
// scores[b] = sum_h relu(q_h . k_b) / sqrt(hd).
// Grid: (n_blocks,1,1)  Block: (hd,1,1).
extern "C" __global__ void qsa_score(
    const float* __restrict__ q,                  // [n_heads, hd]
    const __nv_bfloat16* __restrict__ block_keys, // [*, hd]
    float* __restrict__ scores,                   // [n_blocks]
    const unsigned int n_heads,
    const unsigned int hd
) {
    const unsigned int b = blockIdx.x;
    const unsigned int d = threadIdx.x;

    extern __shared__ float smem[];
    float* red = smem;

    const float k = (float)block_keys[(size_t)b * hd + d];
    float acc = 0.0f;
    for (unsigned int h = 0; h < n_heads; ++h) {
        const float dot = qsa_block_reduce_sum(q[(size_t)h * hd + d] * k, red);
        if (threadIdx.x == 0) acc += fmaxf(dot, 0.0f);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scores[b] = acc * rsqrtf((float)hd);
    }
}

// ── qsa_gather ──
// Pack the selected tokens' K/V rows (NHD paged layout) into contiguous
// scratch: dst slot i holds src position sel[i]. The scratch, viewed through
// an identity block table, IS a valid paged cache for the existing decode
// attention kernel. Grid: (n_sel,1,1)  Block: (256,1,1).
extern "C" __global__ void qsa_gather(
    const __nv_bfloat16* __restrict__ k_cache,    // [blocks, bs, nkv, hd]
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_table,          // logical -> physical
    const int* __restrict__ sel,                  // [n_sel] token positions
    __nv_bfloat16* __restrict__ k_out,            // [n_sel(padded), nkv, hd]
    __nv_bfloat16* __restrict__ v_out,
    const unsigned int block_size,
    const unsigned int nkv,
    const unsigned int hd
) {
    const unsigned int i = blockIdx.x;
    const unsigned int pos = (unsigned int)sel[i];
    const unsigned int row = nkv * hd;
    const unsigned long long page_stride =
        (unsigned long long)block_size * row;
    const unsigned long long src_off =
        (unsigned long long)(unsigned int)block_table[pos / block_size] * page_stride
        + (unsigned long long)(pos % block_size) * row;
    const unsigned long long dst_off = (unsigned long long)i * row;
    for (unsigned int e = threadIdx.x; e < row; e += blockDim.x) {
        k_out[dst_off + e] = k_cache[src_off + e];
        v_out[dst_off + e] = v_cache[src_off + e];
    }
}


// ──────────────────── stage 2: per-query PREFILL selection ────────────────────
//
// Selectivity is monotone in position: every chunk row at global pos >= 2051
// needs its own top-512-block set. Rows are processed as a contiguous range
// [first_pos, first_pos + n_rows); per row the score matrix is masked at the
// row's own complete-block count, host top-k builds a 512-entry block list,
// and qsa_prefill_attn OVERWRITES that row's attention context (pre-gate,
// pre-o_proj) with attention over exactly the selected set — read straight
// from the paged KV cache, so the dense flash pass it replaces needs no
// changes.

// Per-row q prep: RMSNorm*(1+w) + partial rope at pos = first_pos + row.
// qk rows are the indexer projection [rows, (n_heads+1)*hd]; q is the head-
// concatenated prefix of each row. Grid: (rows, n_heads)  Block: (hd,1,1).
extern "C" __global__ void qsa_qprep_rows(
    const __nv_bfloat16* __restrict__ qk,       // [rows, qkw]
    const __nv_bfloat16* __restrict__ q_norm_w, // [hd]
    float* __restrict__ q_out,                  // [rows, n_heads, hd]
    const unsigned int first_pos,
    const unsigned int qkw,
    const unsigned int n_heads,
    const unsigned int hd,
    const unsigned int rot,
    const float theta,
    const float eps
) {
    const unsigned int r = blockIdx.x;
    const unsigned int hh = blockIdx.y;
    const unsigned int d = threadIdx.x;
    const unsigned int pos = first_pos + r;

    extern __shared__ float smem[];
    float* stage = smem;
    float* red = smem + hd;

    const float x = (float)qk[(size_t)r * qkw + (size_t)hh * hd + d];
    const float sq = qsa_block_reduce_sum(x * x, red);
    const float rms = rsqrtf(sq / (float)hd + eps);
    stage[d] = x * rms * (1.0f + (float)q_norm_w[d]);
    __syncthreads();

    float* out = q_out + ((size_t)r * n_heads + hh) * hd;
    if (d < rot) {
        const unsigned int half = rot >> 1;
        const unsigned int j = (d < half) ? d : d - half;
        const double inv_freq = exp(-2.0 * (double)j / (double)rot * log((double)theta));
        double s, c;
        sincos((double)pos * inv_freq, &s, &c);
        const float x1 = stage[j];
        const float x2 = stage[j + half];
        out[d] = (d < half) ? (x1 * (float)c - x2 * (float)s)
                            : (x2 * (float)c + x1 * (float)s);
    } else {
        out[d] = stage[d];
    }
}

// Per-row block scores. scores[r, b] = sum_h relu(q[r,h] . k_b)/sqrt(hd) for
// b < complete(row), -inf otherwise (host top-k then never picks it).
// Grid: (rows, n_blocks_max)  Block: (hd,1,1).
extern "C" __global__ void qsa_score_rows(
    const float* __restrict__ q,                // [rows, n_heads, hd]
    const __nv_bfloat16* __restrict__ block_keys,
    float* __restrict__ scores,                 // [rows, score_stride]
    const unsigned int first_pos,
    const unsigned int score_stride,
    const unsigned int ratio,
    const unsigned int n_heads,
    const unsigned int hd
) {
    const unsigned int r = blockIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int d = threadIdx.x;
    const unsigned int complete = (first_pos + r + 1) / ratio;
    float* out = scores + (size_t)r * score_stride + b;
    if (b >= complete) {
        if (d == 0) *out = -1e30f;
        return;
    }

    extern __shared__ float smem[];
    float* red = smem;

    const float k = (float)block_keys[(size_t)b * hd + d];
    const float* qr = q + (size_t)r * n_heads * hd;
    float acc = 0.0f;
    for (unsigned int hh = 0; hh < n_heads; ++hh) {
        const float dot = qsa_block_reduce_sum(qr[(size_t)hh * hd + d] * k, red);
        if (d == 0) acc += fmaxf(dot, 0.0f);
        __syncthreads();
    }
    if (d == 0) *out = acc * rsqrtf((float)hd);
}

// Block-tiled `qsa_score_rows`. Same scores, bit for bit: each (row, head,
// block) dot is still one 128-thread `qsa_block_reduce_sum` over the same
// operands in the same tree, and the per-head relu/accumulate order is
// unchanged. The ONLY difference is how much work one CTA does.
//
// The untiled kernel launches one CTA per (row, block) — at 16k depth that is
// 2048 rows x 4096 blocks x 4 slabs = 33.5M CTAs per attention layer per chunk,
// and each one re-reads that row's whole query tile (n_heads*hd f32 = 2 KB)
// to consume a single 256-byte block key. Query traffic outweighs key traffic
// 8:1 and is pure redundancy: the query is invariant across the block axis.
// Staging it in shared memory once and looping QSA_SCORE_TB blocks divides
// that term by QSA_SCORE_TB.
//
// Grid: (rows, ceil(n_blocks_max / QSA_SCORE_TB))  Block: (hd,1,1).
#define QSA_SCORE_TB 16
extern "C" __global__ void qsa_score_rows_tiled(
    const float* __restrict__ q,                // [rows, n_heads, hd]
    const __nv_bfloat16* __restrict__ block_keys,
    float* __restrict__ scores,                 // [rows, score_stride]
    const unsigned int first_pos,
    const unsigned int score_stride,
    const unsigned int ratio,
    const unsigned int n_heads,
    const unsigned int hd,
    const unsigned int n_blocks_max
) {
    const unsigned int r = blockIdx.x;
    const unsigned int b0 = blockIdx.y * QSA_SCORE_TB;
    const unsigned int d = threadIdx.x;
    const unsigned int complete = (first_pos + r + 1) / ratio;
    float* row_out = scores + (size_t)r * score_stride;

    // [red(32) | q_tile(n_heads*hd)] — `red` first so `qsa_block_reduce_sum`
    // keeps the exact slab it uses in the untiled kernel.
    extern __shared__ float smem[];
    float* red = smem;
    float* q_tile = smem + 32;

    const float* qr = q + (size_t)r * n_heads * hd;
    for (unsigned int hh = 0; hh < n_heads; ++hh) {
        q_tile[hh * hd + d] = qr[(size_t)hh * hd + d];
    }
    __syncthreads();

    for (unsigned int t = 0; t < QSA_SCORE_TB; ++t) {
        const unsigned int b = b0 + t;
        // Uniform across the CTA (b0, t, and the bounds are all block-wide),
        // so no __syncthreads inside qsa_block_reduce_sum is ever reached by
        // a divergent subset.
        if (b >= n_blocks_max) {
            return;
        }
        if (b >= complete) {
            // Every later b is masked too, but keep writing: the untiled
            // kernel fills [complete, n_blocks_max) with the sentinel and the
            // consumer's scan bound relies on it.
            if (d == 0) {
                row_out[b] = -1e30f;
            }
            continue;
        }
        const float k = (float)block_keys[(size_t)b * hd + d];
        float acc = 0.0f;
        for (unsigned int hh = 0; hh < n_heads; ++hh) {
            const float dot = qsa_block_reduce_sum(q_tile[hh * hd + d] * k, red);
            if (d == 0) {
                acc += fmaxf(dot, 0.0f);
            }
            __syncthreads();
        }
        if (d == 0) {
            row_out[b] = acc * rsqrtf((float)hd);
        }
    }
}

// ─── on-device per-row top-k over the score matrix ───
//
// Replaces a synchronous D2H of the WHOLE score matrix (rows x score_stride
// f32 — 67 MB per attention layer at 16k depth) plus a multi-threaded host
// quickselect. The host path drains the stream once per slab per layer, which
// is why qsa_select was ~80% of attention-layer time post-Phase-1.
//
// The output must be the EXACT same list, in the EXACT same order, as
// `host_topk_blocks`: the selected set feeds attention (a different set is a
// numerics change) and `qsa_prefill_attn` runs an ONLINE softmax over the list
// in order (a different order is an accumulation-order change). The host
// comparator is score DESC, then block index ASC.
//
// That total order is exactly a descending sort of the 64-bit key
//     (float_bits(score) << 32) | ~index
// because every score here is a sum of relu() terms times rsqrt(hd), hence
// >= 0, where __float_as_uint is monotonic; and ~index in the low half breaks
// every tie the way the host does while making the key UNIQUE — which is what
// lets a threshold select return exactly `topk` entries with no tie
// adjudication. Masked lanes (score = -1e30f, sign bit set, NOT monotonic) are
// never read: the scan stops at `complete`, the same bound the host used.
//
// Algorithm per row, one CTA: 8-bit-digit radix select from the top down to
// find the topk-th largest key exactly, gather the `topk` keys at or above it,
// then bitonic-sort them in shared memory and emit descending. The select
// stops early as soon as a digit bucket is consumed whole, which is the
// no-ties-at-the-boundary case.
//
// Grid: (rows,1,1)  Block: (256,1,1)
// Shared: topk*8 bytes of key buffer + QSA_TOPK_HIST_U32*4 bytes of histogram.
#define QSA_TOPK_THREADS 256
// 256 digit counters + digit / remaining / done / gather-counter scratch.
#define QSA_TOPK_HIST_U32 260

__device__ __forceinline__ unsigned long long qsa_topk_key(float s, unsigned int i) {
    return ((unsigned long long)__float_as_uint(s) << 32) | (unsigned long long)(~i);
}

extern "C" __global__ void qsa_topk_rows(
    const float* __restrict__ scores,  // [rows, score_stride] f32, row-major
    int* __restrict__ lists,           // [rows, topk] i32 block ids, score DESC
    const unsigned int first_pos,
    const unsigned int score_stride,
    const unsigned int ratio,
    const unsigned int topk
) {
    const unsigned int r = blockIdx.x;
    // Identical to the host's per-row bound: the number of COMPLETE blocks
    // behind this row. Guaranteed > topk in prefill (first_pos >= 2051).
    const unsigned int n = (first_pos + r + 1) / ratio;
    const float* __restrict__ row = scores + (size_t)r * score_stride;
    int* __restrict__ out = lists + (size_t)r * topk;

    extern __shared__ unsigned long long qsa_topk_smem[];
    unsigned long long* buf = qsa_topk_smem;                      // [topk]
    unsigned int* hist = (unsigned int*)(qsa_topk_smem + topk);   // [260]
    // hist[256] = chosen digit, [257] = how many still wanted from it,
    // [258] = bucket consumed whole (select can stop), [259] = gather counter.

    unsigned long long prefix = 0;  // resolved high bits of the threshold key
    unsigned long long mask = 0;    // which bits of `prefix` are resolved
    unsigned int remaining = topk;

    for (int shift = 56; shift >= 0; shift -= 8) {
        for (unsigned int i = threadIdx.x; i < 256; i += blockDim.x) {
            hist[i] = 0u;
        }
        __syncthreads();
        for (unsigned int b = threadIdx.x; b < n; b += blockDim.x) {
            const unsigned long long k = qsa_topk_key(row[b], b);
            if ((k & mask) == prefix) {
                atomicAdd(&hist[(unsigned int)((k >> shift) & 0xFFull)], 1u);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int rem = remaining;
            int d = 255;
            for (; d > 0; --d) {
                if (hist[d] >= rem) {
                    break;
                }
                rem -= hist[d];
            }
            hist[256] = (unsigned int)d;
            hist[257] = rem;
            hist[258] = (hist[d] == rem) ? 1u : 0u;
        }
        __syncthreads();
        prefix |= ((unsigned long long)hist[256]) << shift;
        mask |= 0xFFull << shift;
        remaining = hist[257];
        const unsigned int done = hist[258];
        __syncthreads();
        if (done) {
            // Every key in this bucket is wanted, so the bucket's lower bound
            // (prefix with the unresolved low bits zero) is already a correct
            // threshold — refining it further cannot change the selected set.
            break;
        }
    }

    // Keys are unique, so {k >= prefix} holds exactly `topk` of them. Seed the
    // buffer with key(0.0, block 0) so a short gather (unreachable given
    // n > topk, but cheap to make safe) emits a valid block id rather than
    // whatever the last slab left behind.
    for (unsigned int i = threadIdx.x; i < topk; i += blockDim.x) {
        buf[i] = qsa_topk_key(0.0f, 0u);
    }
    if (threadIdx.x == 0) {
        hist[259] = 0u;
    }
    __syncthreads();
    for (unsigned int b = threadIdx.x; b < n; b += blockDim.x) {
        const unsigned long long k = qsa_topk_key(row[b], b);
        if (k >= prefix) {
            const unsigned int slot = atomicAdd(&hist[259], 1u);
            if (slot < topk) {
                buf[slot] = k;
            }
        }
    }
    __syncthreads();

    // Bitonic sort ASCENDING (topk is a power of two — launcher-enforced),
    // then emit reversed for the descending order the host produced.
    for (unsigned int kk = 2; kk <= topk; kk <<= 1) {
        for (unsigned int j = kk >> 1; j > 0; j >>= 1) {
            for (unsigned int i = threadIdx.x; i < topk; i += blockDim.x) {
                const unsigned int ixj = i ^ j;
                if (ixj > i) {
                    const bool up = ((i & kk) == 0u);
                    const unsigned long long a = buf[i];
                    const unsigned long long b2 = buf[ixj];
                    if ((a > b2) == up) {
                        buf[i] = b2;
                        buf[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (unsigned int i = threadIdx.x; i < topk; i += blockDim.x) {
        const unsigned int idx =
            ~((unsigned int)(buf[topk - 1u - i] & 0xFFFFFFFFull));
        out[i] = (int)idx;
    }
}

// Attention over EXACTLY the selected set for one (row, q-head): the listed
// `topk` blocks (ratio tokens each) plus the incomplete tail
// [complete*ratio, pos]. K/V come straight from the paged cache; the output
// OVERWRITES that row's context in attn_out (pre-gate, pre-o_proj), so the
// surrounding dense path needs no other change. Softmax is order-invariant
// and rope is baked into cached K, so this equals the reference mask.
// Grid: (rows, nq)  Block: (256,1,1) = 8 warps, warp-striped online softmax.
#define QSA_PA_WARPS 8
// Heads per CTA pass. With nkv=2 and nq=24, the old one-CTA-per-(row,head)
// grid re-streamed each GQA group's K/V rows 12x; at depths where the
// selected working set outruns L2 that redundancy hits DRAM and the kernel
// doubles (measured: qsa_select 6.7 s -> 13.8 s per chunk from depth 16k to
// 24k). Processing QSA_PA_HPP heads per K/V pass divides that traffic by
// HPP while keeping every head's arithmetic bit-identical: per head the
// token order per warp-stripe, the shfl reduction tree, the online-softmax
// update chain and the 8-way stripe merge are exactly the old kernel's —
// heads merely interleave in the instruction stream, and heads never
// interact numerically. Requires (nq / nkv) % QSA_PA_HPP == 0 so one CTA
// never straddles two KV heads (24/2 = 12, HPP 4 divides it); the launcher
// enforces this and grid.y = nq / QSA_PA_HPP.
#define QSA_PA_HPP 4
extern "C" __global__ void qsa_prefill_attn(
    const __nv_bfloat16* __restrict__ q,        // [rows, nq, hd] (roped)
    const __nv_bfloat16* __restrict__ k_cache,  // paged NHD
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ lists,              // [rows, topk] block ids
    __nv_bfloat16* __restrict__ attn_out,       // [rows, nq, hd]
    const unsigned int first_pos,
    const unsigned int topk,
    const unsigned int ratio,
    const unsigned int block_size,
    const unsigned int nq,
    const unsigned int nkv,
    const unsigned int hd,
    const float inv_sqrt_d
) {
    const unsigned int r = blockIdx.x;
    const unsigned int qh0 = blockIdx.y * QSA_PA_HPP;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int pos = first_pos + r;
    const unsigned int complete = (pos + 1) / ratio;
    const unsigned int tail = (pos + 1) - complete * ratio;
    const unsigned int n_tok = topk * ratio + tail;
    const unsigned int kvh = qh0 / (nq / nkv);  // all HPP heads share it
    const unsigned int row_elems = nkv * hd;
    const unsigned long long page_stride = (unsigned long long)block_size * row_elems;
    const unsigned int vec = hd / 32;           // elems per lane (8 at hd=256)

    extern __shared__ float smem[];
    // Per-warp partials for ONE head at a time (heads merge sequentially,
    // reusing the same slab): [warps][hd] acc, then [warps] m, [warps] l.
    float* acc_w = smem;                        // [QSA_PA_WARPS * hd]
    float* m_w = smem + QSA_PA_WARPS * hd;      // [QSA_PA_WARPS]
    float* l_w = m_w + QSA_PA_WARPS;            // [QSA_PA_WARPS]

    // q slices for the HPP heads, staged per lane.
    float qreg[QSA_PA_HPP][8];
    #pragma unroll
    for (unsigned int hh = 0; hh < QSA_PA_HPP; ++hh) {
        const __nv_bfloat16* qrow = q + ((size_t)r * nq + qh0 + hh) * hd;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            qreg[hh][e] = (e < vec) ? (float)qrow[lane * vec + e] : 0.0f;
        }
    }

    float m[QSA_PA_HPP], l[QSA_PA_HPP], acc[QSA_PA_HPP][8];
    #pragma unroll
    for (unsigned int hh = 0; hh < QSA_PA_HPP; ++hh) {
        m[hh] = -1e30f; l[hh] = 0.0f;
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) acc[hh][e] = 0.0f;
    }

    // Token loop, software-pipelined: LPDDR5X LATENCY (not bandwidth) is the
    // measured wall — the per-token chain (list -> block_table -> K row) is
    // two dependent global loads deep, and the online softmax serializes the
    // math across a warp's tokens, so unpipelined every iteration eats a full
    // DRAM round trip. Stage iteration t+WARPS's K/V rows (the lane's 16-byte
    // slice, one uint4 each) while t's math runs. bf16->f32 conversion is
    // exact and the per-token op order is untouched, so the output stays
    // bit-identical to the unpipelined kernel.
    const int* my_list = lists + (size_t)r * topk;
    auto tok_off = [&](unsigned int t) -> unsigned long long {
        unsigned int tok;
        if (t < topk * ratio) {
            tok = (unsigned int)my_list[t / ratio] * ratio + (t % ratio);
        } else {
            tok = complete * ratio + (t - topk * ratio);
        }
        return (unsigned long long)(unsigned int)block_table[tok / block_size] * page_stride
            + (unsigned long long)(tok % block_size) * row_elems
            + (unsigned long long)kvh * hd;
    };
    uint4 k_cur = make_uint4(0, 0, 0, 0), v_cur = make_uint4(0, 0, 0, 0);
    if (warp < n_tok) {
        const unsigned long long off0 = tok_off(warp);
        k_cur = *(const uint4*)(k_cache + off0 + lane * vec);
        v_cur = *(const uint4*)(v_cache + off0 + lane * vec);
    }
    for (unsigned int t = warp; t < n_tok; t += QSA_PA_WARPS) {
        // Issue next iteration's loads ahead of this iteration's math.
        uint4 k_nxt = make_uint4(0, 0, 0, 0), v_nxt = make_uint4(0, 0, 0, 0);
        const unsigned int t_next = t + QSA_PA_WARPS;
        if (t_next < n_tok) {
            const unsigned long long off_n = tok_off(t_next);
            k_nxt = *(const uint4*)(k_cache + off_n + lane * vec);
            v_nxt = *(const uint4*)(v_cache + off_n + lane * vec);
        }
        const __nv_bfloat16* krow = (const __nv_bfloat16*)&k_cur;
        // One K row read serves all HPP heads.
        float kv_elem[8];
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            kv_elem[e] = (e < vec) ? (float)krow[e] : 0.0f;
        }
        float dot[QSA_PA_HPP];
        #pragma unroll
        for (unsigned int hh = 0; hh < QSA_PA_HPP; ++hh) {
            float d = 0.0f;
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) {
                if (e < vec) d += qreg[hh][e] * kv_elem[e];
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) d += __shfl_down_sync(0xFFFFFFFFu, d, o);
            dot[hh] = __shfl_sync(0xFFFFFFFFu, d, 0) * inv_sqrt_d;
        }
        const __nv_bfloat16* vrow = (const __nv_bfloat16*)&v_cur;
        // One V row read serves all HPP heads.
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            kv_elem[e] = (e < vec) ? (float)vrow[e] : 0.0f;
        }
        #pragma unroll
        for (unsigned int hh = 0; hh < QSA_PA_HPP; ++hh) {
            const float m_new = fmaxf(m[hh], dot[hh]);
            const float scale = __expf(m[hh] - m_new);
            const float p = __expf(dot[hh] - m_new);
            l[hh] = l[hh] * scale + p;
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) {
                if (e < vec) acc[hh][e] = acc[hh][e] * scale + p * kv_elem[e];
            }
            m[hh] = m_new;
        }
        k_cur = k_nxt;
        v_cur = v_nxt;
    }

    // Merge one head at a time through the shared slab (barrier-separated so
    // the slab can be reused); warp 0 folds the 8 stripes in warp order —
    // the same merge the old kernel did per head.
    for (unsigned int hh = 0; hh < QSA_PA_HPP; ++hh) {
        #pragma unroll
        for (unsigned int e = 0; e < 8; ++e) {
            if (e < vec) acc_w[warp * hd + lane * vec + e] = acc[hh][e];
        }
        if (lane == 0) { m_w[warp] = m[hh]; l_w[warp] = l[hh]; }
        __syncthreads();

        if (warp == 0) {
            float m_tot = -1e30f;
            for (unsigned int w = 0; w < QSA_PA_WARPS; ++w) m_tot = fmaxf(m_tot, m_w[w]);
            float l_tot = 0.0f;
            float out[8];
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) out[e] = 0.0f;
            for (unsigned int w = 0; w < QSA_PA_WARPS; ++w) {
                const float s = __expf(m_w[w] - m_tot);
                l_tot += l_w[w] * s;
                #pragma unroll
                for (unsigned int e = 0; e < 8; ++e) {
                    if (e < vec) out[e] += acc_w[w * hd + lane * vec + e] * s;
                }
            }
            const float inv_l = (l_tot > 0.0f) ? 1.0f / l_tot : 0.0f;
            __nv_bfloat16* orow = attn_out + ((size_t)r * nq + qh0 + hh) * hd;
            #pragma unroll
            for (unsigned int e = 0; e < 8; ++e) {
                if (e < vec) orow[lane * vec + e] = __float2bfloat16(out[e] * inv_l);
            }
        }
        __syncthreads();
    }
}
