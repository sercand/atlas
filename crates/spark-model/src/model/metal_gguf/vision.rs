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
        let patch_t = store.get(&p("patch_embed.proj.weight"))?;
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
            blocks.push(VisionBlockW {
                norm1_w: b("norm1.weight")?,
                norm1_b: b("norm1.bias")?,
                norm2_w: b("norm2.weight")?,
                norm2_b: b("norm2.bias")?,
                qkv_w: b("attn.qkv.weight")?,
                qkv_b: b("attn.qkv.bias")?,
                proj_w: b("attn.proj.weight")?,
                proj_b: b("attn.proj.bias")?,
                fc1_w: b("mlp.linear_fc1.weight")?,
                fc1_b: b("mlp.linear_fc1.bias")?,
                fc2_w: b("mlp.linear_fc2.weight")?,
                fc2_b: b("mlp.linear_fc2.bias")?,
            });
        }
        let pad_token_id = if vc.image_pad_token_id != 0 {
            vc.image_pad_token_id
        } else {
            151655
        };
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
            merger_fc1_w: get("merger.linear_fc1.weight")?,
            merger_fc1_b: get("merger.linear_fc1.bias")?,
            merger_fc2_w: get("merger.linear_fc2.weight")?,
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

    fn gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        x: DevicePtr,
        w: DevicePtr,
        y: DevicePtr,
    ) -> Result<()> {
        let kern = self.gpu.kernel("dense_gemm_bf16", "dense_gemm_bf16")?;
        self.gpu.launch_typed(
            kern,
            [(n as u32).div_ceil(16), (m as u32).div_ceil(16), 1],
            [16, 16, 1],
            0,
            self.stream,
            &[
                KernelArg::Bytes(&(m as u32).to_le_bytes()),
                KernelArg::Bytes(&(n as u32).to_le_bytes()),
                KernelArg::Bytes(&(k as u32).to_le_bytes()),
                KernelArg::Buffer(x),
                KernelArg::Buffer(w),
                KernelArg::Buffer(y),
            ],
        )
    }

    fn bias(&self, rows: usize, cols: usize, bias: DevicePtr, x: DevicePtr) -> Result<()> {
        let k = self.gpu.kernel("bias_add_rows", "bias_add_rows")?;
        let n = (rows * cols) as u32;
        self.gpu.launch_typed(
            k,
            [n.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            self.stream,
            &[
                KernelArg::Bytes(&(rows as u32).to_le_bytes()),
                KernelArg::Bytes(&(cols as u32).to_le_bytes()),
                KernelArg::Buffer(bias),
                KernelArg::Buffer(x),
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

    fn attention(
        &self,
        v: &MetalVision,
        p: usize,
        q: DevicePtr,
        k: DevicePtr,
        val: DevicePtr,
        out: DevicePtr,
    ) -> Result<()> {
        let kern = self.gpu.kernel("attention_full", "attention_full")?;
        let scale = 1.0f32 / (v.head_dim as f32).sqrt();
        self.gpu.launch_typed(
            kern,
            [(v.heads * p) as u32, 1, 1],
            [32, 1, 1],
            0,
            self.stream,
            &[
                KernelArg::Bytes(&(p as u32).to_le_bytes()),
                KernelArg::Bytes(&(p as u32).to_le_bytes()),
                KernelArg::Bytes(&(v.heads as u32).to_le_bytes()),
                KernelArg::Bytes(&(v.heads as u32).to_le_bytes()),
                KernelArg::Bytes(&(v.head_dim as u32).to_le_bytes()),
                KernelArg::Bytes(&scale.to_le_bytes()),
                KernelArg::Buffer(q),
                KernelArg::Buffer(k),
                KernelArg::Buffer(val),
                KernelArg::Buffer(out),
            ],
        )
    }
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

        // Patch rows f32→bf16 → GEMM (+bias) → + interpolated pos_embed.
        let pix = alloc(p * self.patch_in)?;
        gpu.copy_h2d(&bf16_bytes(pixels), pix)?;
        let h1 = alloc(p * h)?;
        ctx.gemm(p, h, self.patch_in, pix, self.patch_w, h1)?;
        ctx.bias(p, h, self.patch_b, h1)?;
        let pos = alloc(p * h)?;
        gpu.copy_h2d(&self.interp_pos_embed(grid_h, grid_w), pos)?;
        ctx.add(p * h, h1, pos, h1)?;

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
        let wrow = h * 2; // W row-major stride in bytes

        for blk in &self.blocks {
            gpu.copy_d2d_async(h1, resid, p * h * 2, ctx.stream)?;
            ctx.ln(self, p, h1, blk.norm1_w, blk.norm1_b, t1)?;
            // Q/K/V as three GEMMs over the fused weight's row blocks —
            // outputs land contiguous [P, h] each (no interleave split).
            ctx.gemm(p, h, h, t1, blk.qkv_w, q)?;
            ctx.gemm(p, h, h, t1, blk.qkv_w.offset(h * wrow), kbuf)?;
            ctx.gemm(p, h, h, t1, blk.qkv_w.offset(2 * h * wrow), v)?;
            ctx.bias(p, h, blk.qkv_b, q)?;
            ctx.bias(p, h, blk.qkv_b.offset(h * 2), kbuf)?;
            ctx.bias(p, h, blk.qkv_b.offset(2 * h * 2), v)?;
            ctx.vision_rope(self, p, cs, sn, q)?;
            ctx.vision_rope(self, p, cs, sn, kbuf)?;
            ctx.attention(self, p, q, kbuf, v, attn)?;
            ctx.gemm(p, h, h, attn, blk.proj_w, t1)?;
            ctx.bias(p, h, blk.proj_b, t1)?;
            ctx.add(p * h, resid, t1, h1)?;

            gpu.copy_d2d_async(h1, resid, p * h * 2, ctx.stream)?;
            ctx.ln(self, p, h1, blk.norm2_w, blk.norm2_b, t1)?;
            ctx.gemm(p, self.intermediate, h, t1, blk.fc1_w, f)?;
            ctx.bias(p, self.intermediate, blk.fc1_b, f)?;
            ctx.gelu(p * self.intermediate, f)?;
            ctx.gemm(p, h, self.intermediate, f, blk.fc2_w, t1)?;
            ctx.bias(p, h, blk.fc2_b, t1)?;
            ctx.add(p * h, resid, t1, h1)?;
        }

        // Merger: LN (pre-merge) → 2×2 spatial merge (host gather) →
        // fc1 → GELU → fc2 → out rows.
        ctx.ln(self, p, h1, self.merger_norm_w, self.merger_norm_b, t1)?;
        gpu.synchronize(ctx.stream)?;
        let ms = self.merge;
        let (mh, mw) = (grid_h / ms, grid_w / ms);
        let merged_p = mh * mw;
        let merge_dim = ms * ms * h;
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
        ctx.gemm(merged_p, merge_dim, merge_dim, merged, self.merger_fc1_w, g)?;
        ctx.bias(merged_p, merge_dim, self.merger_fc1_b, g)?;
        ctx.gelu(merged_p * merge_dim, g)?;
        ctx.gemm(
            merged_p,
            self.out_hidden,
            merge_dim,
            g,
            self.merger_fc2_w,
            out,
        )?;
        ctx.bias(merged_p, self.out_hidden, self.merger_fc2_b, out)?;
        gpu.synchronize(ctx.stream)?;

        for buf in [
            pix, h1, pos, cs, sn, t1, resid, q, kbuf, v, attn, f, merged, g,
        ] {
            gpu.free(buf)?;
        }
        Ok(merged_p)
    }
}
