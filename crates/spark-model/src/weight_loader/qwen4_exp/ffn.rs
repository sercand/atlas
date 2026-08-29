// SPDX-License-Identifier: AGPL-3.0-only

//! The per-layer MoE block: 512 routed experts, top-10, plus a shared expert
//! and its sigmoid gate.
//!
//! This is `load_moe_qwen35` verbatim. Both the naming
//! (`mlp.gate`, `mlp.shared_expert.{gate,up,down}_proj`,
//! `mlp.shared_expert_gate`, `mlp.experts.{e}.{gate,up,down}_proj`) and the
//! on-disk quantization (standard ModelOpt NVFP4: packed E2M1 `weight`,
//! per-16 E4M3 `weight_scale`, per-tensor F32 `weight_scale_2`) are identical
//! to Qwen3.5/3.6 MoE. The expert COUNT and widths differ — 512 x 640 here
//! against 256 x 512 there — but those come from `config`, not from the
//! loader, so nothing needs re-deriving.
//!
//! The router (`mlp.gate`) is left BF16 and runtime-quantized to NVFP4 only
//! for the non-native-FP8 path, matching qwen35. Its precision matters more
//! than its size: at 512 experts the top-10 weights cluster tightly, and a
//! 4-bit ULP wider than that spread cannot tell them apart.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layers::{FfnComponent, MoeLayer};
use crate::weight_map::{Nvfp4Variant, load_moe_qwen35, quantize_to_nvfp4};

pub(super) fn build_moe(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
) -> Result<FfnComponent> {
    let h = config.hidden_size;
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();

    let weights = load_moe_qwen35(
        store,
        lp,
        config.num_experts,
        gpu,
        config,
        variant,
        absmax_k,
        quantize_k,
        stream,
        false,
    )
    .with_context(|| format!("qwen4_exp: MoE block at {lp}"))?;

    let gate_nvfp4 = Some(quantize_to_nvfp4(
        &weights.gate,
        config.num_experts,
        h,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?);

    let mut moe = MoeLayer::new(weights, config.num_experts, gate_nvfp4, gpu, config)?;

    // CUTLASS grouped NVFP4 prefill (ATLAS_HOLO_MOE_GROUPED_CUTLASS=1):
    // qwen4_exp never builds the Atlas-transposed expert tables (they would
    // double ~68 GB of expert weight on this box), which leaves prefill on the
    // non-transposed BF16-MMA grouped kernel — measured 6.7 TFLOPS, 46% of
    // every chunk on the 125B. The CUTLASS grouped collective reads the
    // checkpoint-native [N,K/2] tables directly, so its only cost is the
    // per-expert swizzled SFB scale copies built here (size-capped inside
    // build_cutlass_grouped_sfb — over the cap it degrades to a no-op, never
    // an OOM). Dispatch is per-call via the moe_grouped_cutlass /
    // moe_grouped_down runtime levers.
    if std::env::var("ATLAS_HOLO_MOE_GROUPED_CUTLASS").ok().as_deref() == Some("1") {
        // Degrade, never fail the boot: without the tables the dispatch just
        // keeps the non-CUTLASS grouped kernels (a no-CUTLASS build bails
        // here on the first pack call).
        if let Err(e) = moe.build_cutlass_grouped_sfb(gpu, config, stream) {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!("qwen4_exp: CUTLASS grouped SFB unavailable at {lp}: {e}");
            }
        }
    }
    Ok(FfnComponent::Moe(moe))
}
