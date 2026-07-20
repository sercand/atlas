// SPDX-License-Identifier: AGPL-3.0-only

//! Metal vision tower for the Qwen3-VL mmproj sidecar (SigLIP-style
//! ViT + 2×2 spatial merger). Mirrors the CUDA `VisionEncoder` pipeline
//! (`layers/vision_encoder/enc_impl/`) through the existing metal
//! kernels: patch embed is a plain GEMM over the preprocessor's
//! `[P, 3·T·16·16]` rows, blocks run LN → QKV → 2D-RoPE SDPA → proj →
//! MLP, and the merger LN + 2×2 merge + MLP produces one `[merged_p,
//! out_hidden]` row per `<|image_pad|>` token. Deepstack is skipped
//! (Bonsai ships `deepstack_visual_indexes = []`).
//!
//! Host-side pieces (UMA): pos-embed bilinear interpolation, the 2D
//! rope cos/sin tables, f32→bf16 pixel conversion, and the 2×2 merge
//! gather — all ports of the CUDA host code in `pos_embed.rs` /
//! `merger.rs`.
//!
//! Perf shape: every GEMM weight converts BF16→F16 IN PLACE at init and
//! runs through the simdgroup-matrix `dense_gemm_f16_bias` (activations
//! cast per GEMM input into a zero-padded device-half image, mirroring
//! the trunk's `q1_0_gemm` contract); attention is the GEMM-based SDPA
//! (`VisionCtx::attention` — per head QKᵀ GEMM → row softmax → P·V
//! GEMM, the CUDA `vit_attention_gemm` port). The naive
//! one-thread-per-cell `dense_gemm_bf16` + serial-softmax
//! `attention_full` tower took ~83 s for a 3072-patch image; this path
//! takes ~2 s.

use anyhow::{Context, Result, bail};
use atlas_core::config::VisionConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg};
use spark_runtime::weights::WeightStore;

/// `attention_full.metal` sizes its threadgroup score buffer to 4096.
const MAX_PATCHES: usize = 4096;
/// SigLIP LayerNorm epsilon.
const LN_EPS: f32 = 1e-6;

struct VisionBlockW {
    norm1_w: DevicePtr,
    norm1_b: DevicePtr,
    norm2_w: DevicePtr,
    norm2_b: DevicePtr,
    qkv_w: DevicePtr,
    qkv_b: DevicePtr,
    proj_w: DevicePtr,
    proj_b: DevicePtr,
    fc1_w: DevicePtr,
    fc1_b: DevicePtr,
    fc2_w: DevicePtr,
    fc2_b: DevicePtr,
}

pub(crate) struct MetalVision {
    hidden: usize,         // 1152
    heads: usize,          // 16
    head_dim: usize,       // 72
    intermediate: usize,   // 4304
    merge: usize,          // 2
    pub out_hidden: usize, // 5120 (= LLM hidden)
    grid_side: usize,      // 48 (sqrt of pos_embed rows)
    pub pad_token_id: u32,
    patch_in: usize, // 1536
    patch_w: DevicePtr,
    patch_b: DevicePtr,
    pos_embed_f32: Vec<f32>, // host copy [grid_side², hidden]
    rope_inv_freq: Vec<f32>, // [head_dim/4]
    blocks: Vec<VisionBlockW>,
    merger_norm_w: DevicePtr,
    merger_norm_b: DevicePtr,
    merger_fc1_w: DevicePtr,
    merger_fc1_b: DevicePtr,
    merger_fc2_w: DevicePtr,
    merger_fc2_b: DevicePtr,
}

/// Encoded rows for one request's images, handed to the prefill splice.
pub(crate) struct VisionRows {
    /// `[rows, out_hidden]` BF16 — one row per `<|image_pad|>` token.
    pub buf: DevicePtr,
    pub rows: usize,
    /// Next row to splice (advances per pad token during prefill).
    pub cursor: usize,
    /// Post-merge (grid_h, grid_w) per image, for MRoPE positions.
    pub grids: Vec<(usize, usize)>,
}

fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }
    out
}

impl MetalVision {
    pub fn from_store(
        store: &WeightStore,
        gpu: &dyn GpuBackend,
        vc: &VisionConfig,
    ) -> Result<Self> {
        let p = |s: &str| format!("model.visual.{s}");
        let get =
            |s: &str| -> Result<DevicePtr> { Ok(store.get(&p(s)).with_context(|| p(s))?.ptr) };
        // GEMM weights additionally register for the in-place BF16→F16
        // convert at the end of construction (biases/norms stay BF16).
        let to_f16: std::cell::RefCell<Vec<(DevicePtr, usize)>> = std::cell::RefCell::new(vec![]);
        let getw = |s: &str| -> Result<DevicePtr> {
            let t = store.get(&p(s)).with_context(|| p(s))?;
            to_f16.borrow_mut().push((t.ptr, t.num_elements()));
            Ok(t.ptr)
        };
        let patch_t = store.get(&p("patch_embed.proj.weight"))?;
        to_f16
            .borrow_mut()
            .push((patch_t.ptr, patch_t.num_elements()));
        let (hidden, patch_in) = (patch_t.shape[0], patch_t.shape[1]);
        let pos_t = store.get(&p("pos_embed.weight"))?;
        let grid_side = (pos_t.shape[0] as f64).sqrt() as usize;
        if grid_side * grid_side != pos_t.shape[0] {
            bail!("pos_embed rows {} not a square grid", pos_t.shape[0]);
        }
        // Host copy of pos_embed for the bilinear interpolation.
        let mut raw = vec![0u8; pos_t.shape[0] * hidden * 2];
        gpu.copy_d2h(pos_t.ptr, &mut raw)?;
        let pos_embed_f32: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();

        let head_dim = hidden / vc.num_heads;
        // Vision rope: head_dim/4 frequencies (row+col split the halves).
        let inv_n = head_dim / 4;
        let rope_inv_freq: Vec<f32> = (0..inv_n)
            .map(|k| 1.0f32 / 10000f32.powf(2.0 * k as f32 / (head_dim / 2) as f32))
            .collect();

        let mut blocks = Vec::with_capacity(vc.depth);
        for i in 0..vc.depth {
            let b = |s: &str| get(&format!("blocks.{i}.{s}"));
            let bw = |s: &str| getw(&format!("blocks.{i}.{s}"));
            blocks.push(VisionBlockW {
                norm1_w: b("norm1.weight")?,
                norm1_b: b("norm1.bias")?,
                norm2_w: b("norm2.weight")?,
                norm2_b: b("norm2.bias")?,
                qkv_w: bw("attn.qkv.weight")?,
                qkv_b: b("attn.qkv.bias")?,
                proj_w: bw("attn.proj.weight")?,
                proj_b: b("attn.proj.bias")?,
                fc1_w: bw("mlp.linear_fc1.weight")?,
                fc1_b: b("mlp.linear_fc1.bias")?,
                fc2_w: bw("mlp.linear_fc2.weight")?,
                fc2_b: b("mlp.linear_fc2.bias")?,
            });
        }
        let pad_token_id = if vc.image_pad_token_id != 0 {
            vc.image_pad_token_id
        } else {
            151655
        };
        let merger_fc1_w = getw("merger.linear_fc1.weight")?;
        let merger_fc2_w = getw("merger.linear_fc2.weight")?;
        // In-place BF16→F16 of every GEMM weight (same byte size; each
        // thread rewrites its own element). One sync covers them all.
        let cvt = gpu.kernel("dense_gemm_f16", "bf16_to_half_inplace")?;
        let stream = gpu.default_stream();
        for (ptr, n) in to_f16.into_inner() {
            gpu.launch_typed(
                cvt,
                [(n as u32).div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&(n as u32).to_le_bytes()),
                    KernelArg::Buffer(ptr),
                ],
            )?;
        }
        gpu.synchronize(stream)?;
        Ok(Self {
            hidden,
            heads: vc.num_heads,
            head_dim,
            intermediate: vc.intermediate_size,
            merge: vc.spatial_merge_size.max(1),
            out_hidden: vc.out_hidden_size,
            grid_side,
            pad_token_id,
            patch_in,
            patch_w: patch_t.ptr,
            patch_b: get("patch_embed.proj.bias")?,
            pos_embed_f32,
            rope_inv_freq,
            blocks,
            merger_norm_w: get("merger.norm.weight")?,
            merger_norm_b: get("merger.norm.bias")?,
            merger_fc1_w,
            merger_fc1_b: get("merger.linear_fc1.bias")?,
            merger_fc2_w,
            merger_fc2_b: get("merger.linear_fc2.bias")?,
        })
    }

    /// Bilinear interpolation of the `grid_side²` pos-embed table down to
    /// `[grid_h·grid_w, hidden]` BF16 host bytes (port of
    /// `resample_pos_embed_into` — torch.linspace endpoint sampling).
    fn interp_pos_embed(&self, grid_h: usize, grid_w: usize) -> Vec<u8> {
        let (h, n) = (self.hidden, self.grid_side);
        let mut out = Vec::with_capacity(grid_h * grid_w * h * 2);
        let denom_h = (grid_h.max(2) - 1) as f32;
        let denom_w = (grid_w.max(2) - 1) as f32;
        for gh in 0..grid_h {
            let fy = if grid_h <= 1 {
                0.0
            } else {
                gh as f32 * (n - 1) as f32 / denom_h
            };
            let y_f = fy.floor() as i32;
            let y_c = (y_f + 1).min(n as i32 - 1);
            let dy = fy - y_f as f32;
            let (y0, y1) = (
                y_f.clamp(0, n as i32 - 1) as usize,
                y_c.clamp(0, n as i32 - 1) as usize,
            );
            for gw in 0..grid_w {
                let fx = if grid_w <= 1 {
                    0.0
                } else {
                    gw as f32 * (n - 1) as f32 / denom_w
                };
                let x_f = fx.floor() as i32;
                let x_c = (x_f + 1).min(n as i32 - 1);
                let dx = fx - x_f as f32;
                let (x0, x1) = (
                    x_f.clamp(0, n as i32 - 1) as usize,
                    x_c.clamp(0, n as i32 - 1) as usize,
                );
                let (w00, w01) = ((1.0 - dy) * (1.0 - dx), (1.0 - dy) * dx);
                let (w10, w11) = (dy * (1.0 - dx), dy * dx);
                let (i00, i01) = ((y0 * n + x0) * h, (y0 * n + x1) * h);
                let (i10, i11) = ((y1 * n + x0) * h, (y1 * n + x1) * h);
                for k in 0..h {
                    let v = w00 * self.pos_embed_f32[i00 + k]
                        + w01 * self.pos_embed_f32[i01 + k]
                        + w10 * self.pos_embed_f32[i10 + k]
                        + w11 * self.pos_embed_f32[i11 + k];
                    out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
                }
            }
        }
        out
    }

    /// Per-patch 2D rope cos/sin tables `[P, head_dim]` BF16 (port of
    /// `build_rope_cossin_into`: `[row_freq; col_freq]` duplicated over
    /// the two halves).
    fn rope_tables(&self, grid_h: usize, grid_w: usize) -> (Vec<u8>, Vec<u8>) {
        let hd = self.head_dim;
        let half = hd / 2;
        let inv_n = self.rope_inv_freq.len();
        let p = grid_h * grid_w;
        let mut cos_v = vec![0f32; p * hd];
        let mut sin_v = vec![0f32; p * hd];
        for gh in 0..grid_h {
            for gw in 0..grid_w {
                let off = (gh * grid_w + gw) * hd;
                for k in 0..inv_n {
                    let rf = gh as f32 * self.rope_inv_freq[k];
                    let cf = gw as f32 * self.rope_inv_freq[k];
                    for (slot, ang) in [(k, rf), (inv_n + k, cf)] {
                        cos_v[off + slot] = ang.cos();
                        sin_v[off + slot] = ang.sin();
                        cos_v[off + half + slot] = ang.cos();
                        sin_v[off + half + slot] = ang.sin();
                    }
                }
            }
        }
        (bf16_bytes(&cos_v), bf16_bytes(&sin_v))
    }
}

/// Per-image scratch + kernel plumbing, split out so `forward_image`
/// stays readable. All launches go through `&dyn GpuBackend`.
pub(crate) struct VisionCtx<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub stream: u64,
}

impl VisionCtx<'_> {
    /// `ATLAS_VISION_DEBUG=1` parity probe: sync, pull `n` BF16 elements
    /// and log their f32 sum. Names/values line up with the tensor sums
    /// `llama-mtmd-debug -p encode` prints for the same mmproj, so the
    /// first diverging stage localizes a numerics bug.
    fn dbg_sum(&self, name: &str, ptr: DevicePtr, n: usize) {
        if !std::env::var("ATLAS_VISION_DEBUG").is_ok_and(|v| v != "0") {
            return;
        }
        let mut raw = vec![0u8; n * 2];
        if self.gpu.synchronize(self.stream).is_err() || self.gpu.copy_d2h(ptr, &mut raw).is_err() {
            return;
        }
        let sum: f64 = raw
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32() as f64)
            .sum();
        tracing::info!("vdbg {name}: sum {sum:.4}");
    }

    fn ln(
        &self,
        v: &MetalVision,
        p: usize,
        x: DevicePtr,
        w: DevicePtr,
        b: DevicePtr,
        out: DevicePtr,
    ) -> Result<()> {
        let k = self.gpu.kernel("layer_norm", "layer_norm")?;
        self.gpu.launch_typed(
            k,
            [p as u32, 1, 1],
            [128, 1, 1],
            0,
            self.stream,
            &[
                KernelArg::Bytes(&(v.hidden as u32).to_le_bytes()),
                KernelArg::Bytes(&LN_EPS.to_le_bytes()),
                KernelArg::Buffer(x),
                KernelArg::Buffer(w),
                KernelArg::Buffer(b),
                KernelArg::Buffer(out),
            ],
        )
    }

    /// Cast `rows × k` BF16 rows at `src` into the shared device-half
    /// GEMM input image `dst` (`[rows_pad, k_pad]`, both tails ZERO —
    /// the GEMM's caller contract). Returns `k_pad`.
    fn cast_pad(&self, rows: usize, k: usize, src: DevicePtr, dst: DevicePtr) -> Result<usize> {
        let kern = self.gpu.kernel("dense_gemm_f16", "bf16_to_half_pad")?;
        let rows_pad = rows.next_multiple_of(128);
        let k_pad = k.next_multiple_of(64);
        let total = (rows_pad * k_pad) as u32;
        self.gpu.launch_typed(
            kern,
            [total.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            self.stream,
            &[
                KernelArg::Bytes(&(rows as u32).to_le_bytes()),
                KernelArg::Bytes(&(k as u32).to_le_bytes()),
                KernelArg::Bytes(&(rows_pad as u32).to_le_bytes()),
                KernelArg::Bytes(&(k_pad as u32).to_le_bytes()),
                KernelArg::Buffer(src),
                KernelArg::Buffer(dst),
            ],
        )?;
        Ok(k_pad)
    }

    /// `y[m, n] = bias[n] + x @ W[n, k]ᵀ` over the pre-cast half input
    /// (`x_half` from [`Self::cast_pad`], row stride `k_pad`).
    #[allow(clippy::too_many_arguments)]
    fn gemm_bias(
        &self,
        m: usize,
        n: usize,
        k: usize,
        k_pad: usize,
        x_half: DevicePtr,
        w: DevicePtr,
        bias: DevicePtr,
        y: DevicePtr,
    ) -> Result<()> {
        let kern = self.gpu.kernel("dense_gemm_f16", "dense_gemm_f16_bias")?;
        let row_tiles = (n as u32).div_ceil(32);
        let t_tiles = (m as u32).div_ceil(128);
        self.gpu.launch_typed(
            kern,
            [row_tiles * t_tiles, 1, 1],
            [128, 1, 1],
            0,
            self.stream,
            &[
                KernelArg::Bytes(&(n as u32).to_le_bytes()),
                KernelArg::Bytes(&(k as u32).to_le_bytes()),
                KernelArg::Bytes(&(m as u32).to_le_bytes()),
                KernelArg::Bytes(&(k_pad as u32).to_le_bytes()),
                KernelArg::Buffer(w),
                KernelArg::Buffer(x_half),
                KernelArg::Buffer(bias),
                KernelArg::Buffer(y),
            ],
        )
    }

    fn add(&self, n: usize, a: DevicePtr, b: DevicePtr, out: DevicePtr) -> Result<()> {
        let k = self.gpu.kernel("bf16_add", "bf16_add")?;
        self.gpu.launch_typed(
            k,
            [(n as u32).div_ceil(64), 1, 1],
            [64, 1, 1],
            0,
            self.stream,
            &[
                KernelArg::Bytes(&(n as u32).to_le_bytes()),
                KernelArg::Buffer(a),
                KernelArg::Buffer(b),
                KernelArg::Buffer(out),
            ],
        )
    }

    fn gelu(&self, n: usize, x: DevicePtr) -> Result<()> {
        let k = self.gpu.kernel("gelu", "gelu")?;
        self.gpu.launch_typed(
            k,
            [(n as u32).div_ceil(64), 1, 1],
            [64, 1, 1],
            0,
            self.stream,
            &[
                KernelArg::Bytes(&(n as u32).to_le_bytes()),
                KernelArg::Buffer(x),
                KernelArg::Buffer(x),
            ],
        )
    }

    fn vision_rope(
        &self,
        v: &MetalVision,
        p: usize,
        cs: DevicePtr,
        sn: DevicePtr,
        x: DevicePtr,
    ) -> Result<()> {
        let k = self.gpu.kernel("vision_rope_apply", "vision_rope_apply")?;
        self.gpu.launch_typed(
            k,
            [
                ((v.head_dim / 2) as u32).div_ceil(32),
                v.heads as u32,
                p as u32,
            ],
            [32, 1, 1],
            0,
            self.stream,
            &[
                KernelArg::Bytes(&(p as u32).to_le_bytes()),
                KernelArg::Bytes(&(v.heads as u32).to_le_bytes()),
                KernelArg::Bytes(&(v.head_dim as u32).to_le_bytes()),
                KernelArg::Buffer(cs),
                KernelArg::Buffer(sn),
                KernelArg::Buffer(x),
            ],
        )
    }

    /// Head-slice cast into a zero-padded half image (see
    /// `bf16_to_half_pad_strided`). `src` is pre-offset to the head.
    #[allow(clippy::too_many_arguments)]
    fn cast_head(
        &self,
        rows: usize,
        k: usize,
        rows_pad: usize,
        k_pad: usize,
        src_stride: usize,
        src: DevicePtr,
        dst: DevicePtr,
    ) -> Result<()> {
        let kern = self.gpu.kernel("dense_gemm_f16", "bf16_to_half_pad_strided")?;
        let total = (rows_pad * k_pad) as u32;
        self.gpu.launch_typed(
            kern,
            [total.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            self.stream,
            &[
                KernelArg::Bytes(&(rows as u32).to_le_bytes()),
                KernelArg::Bytes(&(k as u32).to_le_bytes()),
                KernelArg::Bytes(&(rows_pad as u32).to_le_bytes()),
                KernelArg::Bytes(&(k_pad as u32).to_le_bytes()),
                KernelArg::Bytes(&(src_stride as u32).to_le_bytes()),
                KernelArg::Buffer(src),
                KernelArg::Buffer(dst),
            ],
        )
    }

    /// GEMM-based SDPA (the CUDA `vit_attention_gemm` port): per head,
    /// scores = Q·Kᵀ through the simdgroup-matrix GEMM (f32 out), row
    /// softmax (scale folded, half out, zero-padded), then P·V through
    /// the direct-B GEMM back into the head's slice of `out`. One-query-
    /// per-threadgroup attention kernels re-stream all of K/V per query
    /// (~630 ms/layer at P=3072); this runs the same math as two ~82%-
    /// FMA-peak GEMMs (~30 ms/layer).
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        v: &MetalVision,
        p: usize,
        sc: &AttnScratch,
        q: DevicePtr,
        k: DevicePtr,
        val: DevicePtr,
        out: DevicePtr,
    ) -> Result<()> {
        let hd = v.head_dim;
        let hidden = v.hidden;
        let p_pad = p.next_multiple_of(128);
        let s_pad = p.next_multiple_of(64);
        let n_pad = hd.next_multiple_of(32);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let gemm1 = self.gpu.kernel("dense_gemm_f16", "dense_gemm_f16_f32out")?;
        let softmax = self.gpu.kernel("dense_gemm_f16", "vit_softmax_half")?;
        let gemm2 = self.gpu.kernel("dense_gemm_f16", "dense_gemm_f16_nt")?;
        for h in 0..v.heads {
            let off = h * hd * 2;
            self.cast_head(p, hd, p_pad, 128, hidden, q.offset(off), sc.xq)?;
            self.cast_head(p, hd, p, hd, hidden, k.offset(off), sc.wk)?;
            self.gpu.launch_typed(
                gemm1,
                [(p as u32).div_ceil(32) * (p as u32).div_ceil(128), 1, 1],
                [128, 1, 1],
                0,
                self.stream,
                &[
                    KernelArg::Bytes(&(p as u32).to_le_bytes()),
                    KernelArg::Bytes(&(hd as u32).to_le_bytes()),
                    KernelArg::Bytes(&(p as u32).to_le_bytes()),
                    KernelArg::Bytes(&128u32.to_le_bytes()),
                    KernelArg::Buffer(sc.wk),
                    KernelArg::Buffer(sc.xq),
                    KernelArg::Buffer(sc.scores),
                ],
            )?;
            self.gpu.launch_typed(
                softmax,
                [p_pad as u32, 1, 1],
                [128, 1, 1],
                0,
                self.stream,
                &[
                    KernelArg::Bytes(&(p as u32).to_le_bytes()),
                    KernelArg::Bytes(&(p as u32).to_le_bytes()),
                    KernelArg::Bytes(&(s_pad as u32).to_le_bytes()),
                    KernelArg::Bytes(&scale.to_le_bytes()),
                    KernelArg::Buffer(sc.scores),
                    KernelArg::Buffer(sc.probs),
                ],
            )?;
            self.cast_head(p, hd, s_pad, n_pad, hidden, val.offset(off), sc.wv)?;
            self.gpu.launch_typed(
                gemm2,
                [(n_pad as u32).div_ceil(32) * (p as u32).div_ceil(128), 1, 1],
                [128, 1, 1],
                0,
                self.stream,
                &[
                    KernelArg::Bytes(&(hd as u32).to_le_bytes()),
                    KernelArg::Bytes(&(p as u32).to_le_bytes()),
                    KernelArg::Bytes(&(s_pad as u32).to_le_bytes()),
                    KernelArg::Bytes(&(n_pad as u32).to_le_bytes()),
                    KernelArg::Bytes(&(hidden as u32).to_le_bytes()),
                    KernelArg::Buffer(sc.wv),
                    KernelArg::Buffer(sc.probs),
                    KernelArg::Buffer(out.offset(off)),
                ],
            )?;
        }
        Ok(())
    }
}

/// Per-image device scratch for the GEMM attention path.
struct AttnScratch {
    xq: DevicePtr,     // [p_pad, 128] half — Q head image
    wk: DevicePtr,     // [p, head_dim] half — K head image (GEMM1 W)
    scores: DevicePtr, // [p, p] f32 — raw Q·Kᵀ
    probs: DevicePtr,  // [p_pad, s_pad] half — softmax rows, zero-padded
    wv: DevicePtr,     // [s_pad, n_pad] half — V head image (GEMM2 B)
}

impl MetalVision {
    /// Encode one image (`pixels` = `[P, patch_in]` f32 from the
    /// preprocessor) and append `merged_p` rows of `[out_hidden]` BF16 at
    /// `out` (which must have room). Returns `merged_p`.
    pub fn forward_image(
        &self,
        ctx: &VisionCtx<'_>,
        pixels: &[f32],
        grid_h: usize,
        grid_w: usize,
        out: DevicePtr,
    ) -> Result<usize> {
        let gpu = ctx.gpu;
        let p = grid_h * grid_w;
        if p > MAX_PATCHES {
            bail!(
                "image has {p} patches; the metal attention_full kernel caps at {MAX_PATCHES} \
                 (lower the image resolution / max_pixels)"
            );
        }
        if pixels.len() != p * self.patch_in {
            bail!("pixel buffer {} != P {p} × {}", pixels.len(), self.patch_in);
        }
        let h = self.hidden;
        let alloc = |n_elems: usize| -> Result<DevicePtr> { gpu.alloc(n_elems * 2) };

        // Shared device-half GEMM input image (`cast_pad` fills it per
        // GEMM input; rows pad to 128, K to 64 — see dense_gemm_f16).
        let pad_t = |t: usize| t.next_multiple_of(128);
        let pad_k = |k: usize| k.next_multiple_of(64);
        let ms = self.merge;
        let (mh, mw) = (grid_h / ms, grid_w / ms);
        let merged_p = mh * mw;
        let merge_dim = ms * ms * h;
        let x_half_elems = [
            pad_t(p) * pad_k(self.patch_in),
            pad_t(p) * pad_k(self.intermediate),
            pad_t(merged_p) * pad_k(merge_dim),
        ]
        .into_iter()
        .max()
        .expect("non-empty");
        let x_half = alloc(x_half_elems)?;

        // Patch rows f32→bf16 → GEMM (+bias) → + interpolated pos_embed.
        let pix = alloc(p * self.patch_in)?;
        gpu.copy_h2d(&bf16_bytes(pixels), pix)?;
        let h1 = alloc(p * h)?;
        let kp = ctx.cast_pad(p, self.patch_in, pix, x_half)?;
        ctx.gemm_bias(p, h, self.patch_in, kp, x_half, self.patch_w, self.patch_b, h1)?;
        ctx.dbg_sum("patch_bias", h1, p * h);
        let pos = alloc(p * h)?;
        gpu.copy_h2d(&self.interp_pos_embed(grid_h, grid_w), pos)?;
        ctx.dbg_sum("pos_interp", pos, p * h);
        ctx.add(p * h, h1, pos, h1)?;
        ctx.dbg_sum("inp_pos_emb", h1, p * h);

        // 2D rope tables.
        let (cos_b, sin_b) = self.rope_tables(grid_h, grid_w);
        let cs = alloc(p * self.head_dim)?;
        let sn = alloc(p * self.head_dim)?;
        gpu.copy_h2d(&cos_b, cs)?;
        gpu.copy_h2d(&sin_b, sn)?;

        // Block scratch.
        let t1 = alloc(p * h)?;
        let resid = alloc(p * h)?;
        let q = alloc(p * h)?;
        let kbuf = alloc(p * h)?;
        let v = alloc(p * h)?;
        let attn = alloc(p * h)?;
        let f = alloc(p * self.intermediate)?;
        let p_pad = p.next_multiple_of(128);
        let s_pad = p.next_multiple_of(64);
        let n_pad = self.head_dim.next_multiple_of(32);
        let attn_sc = AttnScratch {
            xq: alloc(p_pad * 128)?,
            wk: alloc(p * self.head_dim)?,
            scores: gpu.alloc(p * p * 4)?,
            probs: alloc(p_pad * s_pad)?,
            wv: alloc(s_pad * n_pad)?,
        };
        let wrow = h * 2; // W row-major stride in bytes
        let dbg_on = std::env::var("ATLAS_VISION_DEBUG").is_ok_and(|v| v != "0");

        for (bi, blk) in self.blocks.iter().enumerate() {
            let d = |name: &str, ptr: DevicePtr, n: usize| {
                if dbg_on {
                    ctx.dbg_sum(&format!("{name}-{bi}"), ptr, n);
                }
            };
            gpu.copy_d2d_async(h1, resid, p * h * 2, ctx.stream)?;
            ctx.ln(self, p, h1, blk.norm1_w, blk.norm1_b, t1)?;
            d("ln1", t1, p * h);
            // Q/K/V as three GEMMs over the fused weight's row blocks —
            // outputs land contiguous [P, h] each (no interleave split).
            let kp = ctx.cast_pad(p, h, t1, x_half)?;
            ctx.gemm_bias(p, h, h, kp, x_half, blk.qkv_w, blk.qkv_b, q)?;
            ctx.gemm_bias(
                p,
                h,
                h,
                kp,
                x_half,
                blk.qkv_w.offset(h * wrow),
                blk.qkv_b.offset(h * 2),
                kbuf,
            )?;
            ctx.gemm_bias(
                p,
                h,
                h,
                kp,
                x_half,
                blk.qkv_w.offset(2 * h * wrow),
                blk.qkv_b.offset(2 * h * 2),
                v,
            )?;
            d("q_bias", q, p * h);
            d("k_bias", kbuf, p * h);
            d("v_bias", v, p * h);
            ctx.vision_rope(self, p, cs, sn, q)?;
            ctx.vision_rope(self, p, cs, sn, kbuf)?;
            d("q_rope", q, p * h);
            ctx.attention(self, p, &attn_sc, q, kbuf, v, attn)?;
            d("attn", attn, p * h);
            let kp = ctx.cast_pad(p, h, attn, x_half)?;
            ctx.gemm_bias(p, h, h, kp, x_half, blk.proj_w, blk.proj_b, t1)?;
            d("attn_out", t1, p * h);
            ctx.add(p * h, resid, t1, h1)?;
            d("ffn_inp", h1, p * h);

            gpu.copy_d2d_async(h1, resid, p * h * 2, ctx.stream)?;
            ctx.ln(self, p, h1, blk.norm2_w, blk.norm2_b, t1)?;
            let kp = ctx.cast_pad(p, h, t1, x_half)?;
            ctx.gemm_bias(p, self.intermediate, h, kp, x_half, blk.fc1_w, blk.fc1_b, f)?;
            d("ffn_up_b", f, p * self.intermediate);
            ctx.gelu(p * self.intermediate, f)?;
            let kp = ctx.cast_pad(p, self.intermediate, f, x_half)?;
            ctx.gemm_bias(p, h, self.intermediate, kp, x_half, blk.fc2_w, blk.fc2_b, t1)?;
            ctx.add(p * h, resid, t1, h1)?;
            d("layer_out", h1, p * h);
        }

        // Merger: LN (pre-merge) → 2×2 spatial merge (host gather) →
        // fc1 → GELU → fc2 → out rows.
        ctx.ln(self, p, h1, self.merger_norm_w, self.merger_norm_b, t1)?;
        ctx.dbg_sum("merger_ln", t1, p * h);
        gpu.synchronize(ctx.stream)?;
        let mut ln_host = vec![0u8; p * h * 2];
        gpu.copy_d2h(t1, &mut ln_host)?;
        let row = |r: usize, c: usize| -> &[u8] {
            let idx = (r * grid_w + c) * h * 2;
            &ln_host[idx..idx + h * 2]
        };
        let mut merged_host = Vec::with_capacity(merged_p * merge_dim * 2);
        for r in 0..mh {
            for c in 0..mw {
                for dr in 0..ms {
                    for dc in 0..ms {
                        merged_host.extend_from_slice(row(r * ms + dr, c * ms + dc));
                    }
                }
            }
        }
        let merged = alloc(merged_p * merge_dim)?;
        gpu.copy_h2d(&merged_host, merged)?;
        let g = alloc(merged_p * merge_dim)?;
        let kp = ctx.cast_pad(merged_p, merge_dim, merged, x_half)?;
        ctx.gemm_bias(
            merged_p,
            merge_dim,
            merge_dim,
            kp,
            x_half,
            self.merger_fc1_w,
            self.merger_fc1_b,
            g,
        )?;
        ctx.dbg_sum("merger_fc1_b", g, merged_p * merge_dim);
        ctx.gelu(merged_p * merge_dim, g)?;
        let kp = ctx.cast_pad(merged_p, merge_dim, g, x_half)?;
        ctx.gemm_bias(
            merged_p,
            self.out_hidden,
            merge_dim,
            kp,
            x_half,
            self.merger_fc2_w,
            self.merger_fc2_b,
            out,
        )?;
        ctx.dbg_sum("out_rows", out, merged_p * self.out_hidden);
        gpu.synchronize(ctx.stream)?;

        for buf in [
            pix, h1, pos, cs, sn, t1, resid, q, kbuf, v, attn, f, merged, g, x_half, attn_sc.xq,
            attn_sc.wk, attn_sc.scores, attn_sc.probs, attn_sc.wv,
        ] {
            gpu.free(buf)?;
        }
        Ok(merged_p)
    }
}
