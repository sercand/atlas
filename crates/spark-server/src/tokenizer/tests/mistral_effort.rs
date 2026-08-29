// SPDX-License-Identifier: AGPL-3.0-only

//! Mistral-override `reasoning_effort` renders through the PRODUCTION
//! `render_chat` path (the same core the golden Qwen tests prove).
//!
//! Mistral is the only other template family that consumes the
//! `reasoning_effort` Jinja variable, so it pins the OTHER half of the
//! 2026-08-15 unset-default change: Atlas's cross-template fallback moved
//! from `"high"` (which Qwen3.8 escalated to its most expensive `xhigh`
//! directive) to the neutral `"medium"`. Mistral's ladder is binary
//! (`none|high`), so its Atlas-owned override maps `medium` → `high` —
//! keeping the unset Mistral render byte-identical to the pre-change
//! behavior while Qwen3.8 drops to its neutral tier.

use super::super::chat_render::{RenderFlags, render_chat};
use super::super::jinja_helpers;
use serde_json::json;

fn render_mistral(flags: RenderFlags<'_>) -> anyhow::Result<String> {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../jinja-templates/mistral.jinja"
    ))
    .expect("bundled Mistral override template must be present");
    let converted = jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = jinja_helpers::build_jinja_env(&converted).expect("template compiles");
    let messages = [json!({"role": "user", "content": "Hello"})];
    render_chat(&env, &messages, None, flags)
}

const SETTINGS_HIGH: &str = r#"[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]"#;
const SETTINGS_NONE: &str = r#"[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]"#;

/// Unset + thinking on: the "medium" fallback must land on Mistral's
/// standard thinking tier ("high") via the override's mapping — the same
/// bytes the old "high" fallback produced. If this breaks, the fallback
/// in chat_render.rs and the mapping in mistral.jinja have drifted apart.
#[test]
fn unset_effort_thinking_on_renders_high_settings() {
    let r = render_mistral(RenderFlags {
        enable_thinking: true,
        ..Default::default()
    })
    .unwrap();
    assert!(r.contains(SETTINGS_HIGH), "render:\n{r}");
}

/// Explicit "medium" (a client or `--default-chat-template-kwargs`
/// choosing the neutral tier) maps identically to "high" — Mistral has no
/// medium; the neutral tier IS its standard thinking mode.
#[test]
fn explicit_medium_maps_to_high_settings() {
    let r = render_mistral(RenderFlags {
        enable_thinking: true,
        reasoning_effort: Some("medium"),
        ..Default::default()
    })
    .unwrap();
    assert!(r.contains(SETTINGS_HIGH), "render:\n{r}");
}

/// Thinking off must still emit the explicit "none" settings — the
/// medium fallback applies ONLY when thinking is on.
#[test]
fn unset_effort_thinking_off_renders_none_settings() {
    let r = render_mistral(RenderFlags::default()).unwrap();
    assert!(r.contains(SETTINGS_NONE), "render:\n{r}");
}

/// Tiers Mistral does not have stay rejected: an explicit "low" raises
/// in-template (a 400 at the API). Supported-tier vocabulary remains
/// per-model — the ONE exception is the high/xhigh synonym below.
#[test]
fn unsupported_explicit_tiers_still_raise() {
    let err = render_mistral(RenderFlags {
        enable_thinking: true,
        reasoning_effort: Some("low"),
        ..Default::default()
    })
    .unwrap_err();
    assert!(
        format!("{err:#}").contains("reasoning_effort must be either"),
        "expected the Mistral validator to fire, got: {err:#}"
    );
}

/// BEHAVIOUR CHANGE 2026-08-29: `xhigh` used to raise here and now renders
/// as Mistral's `high`.
///
/// `high` and `xhigh` are the same rung spelled two ways, and the templates
/// disagree about which spelling they know: Qwen3.8-Flash-Next accepts only
/// `xhigh`, Mistral accepts only `high`. Before this, `reasoning_effort:
/// "high"` 400'd on Flash-Next — an ordinary OpenAI value rejected because
/// of a spelling. `render_chat` now retries once with the neighbouring
/// spelling when a template raises specifically about the effort value.
///
/// Fixing only the Flash-Next direction would be incoherent: `high` would
/// work on a template that knows only `xhigh` while `xhigh` still 400'd on
/// one that knows only `high`. So the retry is symmetric and a top-tier
/// request lands on whatever the model calls its top tier. No OTHER tier is
/// remapped — see `unsupported_explicit_tiers_still_raise` — so a genuinely
/// unsupported value is still a loud failure, not a silent downgrade.
#[test]
fn xhigh_is_accepted_as_the_synonym_of_mistrals_high() {
    let r = render_mistral(RenderFlags {
        enable_thinking: true,
        reasoning_effort: Some("xhigh"),
        ..Default::default()
    })
    .expect("xhigh must reach Mistral's top thinking tier, not 400");
    assert!(r.contains(SETTINGS_HIGH), "render:\n{r}");
}
