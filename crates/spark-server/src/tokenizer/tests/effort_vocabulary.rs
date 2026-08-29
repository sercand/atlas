// SPDX-License-Identifier: AGPL-3.0-only

//! Reasoning-effort vocabulary divergence between checkpoint templates
//! (2026-08-29).
//!
//! `reasoning_effort: "high"` — an ordinary OpenAI value — returned HTTP 400
//! on `primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8`:
//!
//! ```text
//! Failed to render Jinja chat template: invalid operation: Unexpected
//! reasoning effort high. Supported types are xhigh (default), medium, and low.
//! ```
//!
//! The two templates validate against the SAME tuple but only one of them
//! remaps into it:
//!
//! | template            | `high` |
//! |---------------------|--------|
//! | Qwen3.8-27B         | remapped to `xhigh` by the template itself |
//! | Qwen3.8-Flash-Next  | raises — no remap |
//! | Mistral             | `high` is the only thinking tier; `xhigh` raises |
//!
//! So no static `ReasoningEffort::as_str` table satisfies both directions,
//! and a template cannot be asked what it accepts. `render_chat` renders,
//! and on an effort-specific raise retries ONCE with the neighbouring
//! spelling. These tests pin both directions and, just as importantly, that
//! a genuinely bad value still fails instead of being silently downgraded.

use super::super::chat_render::RenderFlags;
use super::qwen_dense::render_fixture;
use serde_json::json;

const FLASH_NEXT: &str = "qwen3.8-flash-next";
const DENSE_27B: &str = "qwen3.8-27b-unsloth";

/// The sentence the Flash-Next template injects for the top tier. Its
/// presence is how we prove the retry landed on `xhigh` rather than
/// silently dropping the directive.
const XHIGH_SENTENCE: &str = "Reasoning effort is set to xhigh.";
const LOW_SENTENCE: &str = "Reasoning effort is set to low.";

fn msgs() -> Vec<serde_json::Value> {
    vec![json!({"role": "user", "content": "hi"})]
}

fn flags(effort: Option<&str>) -> RenderFlags<'_> {
    RenderFlags {
        enable_thinking: true,
        reasoning_effort: effort,
        ..Default::default()
    }
}

/// The live 400, now a render that succeeds at the equivalent tier.
#[test]
fn flash_next_accepts_high_by_retrying_as_xhigh() {
    let out = render_fixture(FLASH_NEXT, &msgs(), None, flags(Some("high")))
        .expect("`high` must not 400 on a template that lacks the remap");
    assert!(
        out.contains(XHIGH_SENTENCE),
        "the retry must land on the xhigh tier, not drop the directive:\n{out}"
    );
}

/// `xhigh` is this template's native spelling — no retry involved.
#[test]
fn flash_next_accepts_xhigh_directly() {
    let out = render_fixture(FLASH_NEXT, &msgs(), None, flags(Some("xhigh"))).expect("native");
    assert!(out.contains(XHIGH_SENTENCE), "{out}");
}

/// The tiers that need no adaptation must be untouched by the retry path.
#[test]
fn flash_next_low_and_medium_are_unchanged() {
    let low = render_fixture(FLASH_NEXT, &msgs(), None, flags(Some("low"))).expect("low");
    assert!(low.contains(LOW_SENTENCE), "{low}");

    // `medium` is the neutral tier: accepted, and injects NO directive.
    let med = render_fixture(FLASH_NEXT, &msgs(), None, flags(Some("medium"))).expect("medium");
    assert!(!med.contains("Reasoning effort is set to"), "{med}");
}

/// Unset + thinking on resolves to `medium` in `render_chat`, so it must
/// render with no directive and never reach the retry.
#[test]
fn flash_next_unset_effort_is_the_neutral_tier() {
    let out = render_fixture(FLASH_NEXT, &msgs(), None, flags(None)).expect("unset");
    assert!(!out.contains("Reasoning effort is set to"), "{out}");
}

/// The retry remaps ONLY the `high`/`xhigh` synonym pair. A value no
/// template accepts must still fail loudly — a silent downgrade to some
/// working tier would hide a client bug and change billing-relevant
/// behaviour without telling anyone.
#[test]
fn an_unsupported_effort_still_fails() {
    let err = render_fixture(FLASH_NEXT, &msgs(), None, flags(Some("turbo")))
        .expect_err("`turbo` is not a tier on any template");
    let msg = err.to_string();
    assert!(
        msg.contains("reasoning effort") || msg.contains("Unexpected"),
        "the template's own diagnostic must survive: {msg}"
    );
}

/// The 27B template does its own `high` -> `xhigh` remap, so it renders on
/// the FIRST attempt. This pins that the new retry did not disturb it.
#[test]
fn dense_27b_high_is_unaffected() {
    let out = render_fixture(DENSE_27B, &msgs(), None, flags(Some("high")))
        .expect("27B remaps high itself");
    assert!(
        out.contains("Reasoning effort is set to xhigh"),
        "27B must still reach the xhigh tier via its own remap:\n{out}"
    );
}

/// Both spellings must reach the same tier on Flash-Next — that is what
/// makes the retry a spelling fix rather than a behaviour change.
#[test]
fn high_and_xhigh_render_identically_on_flash_next() {
    let a = render_fixture(FLASH_NEXT, &msgs(), None, flags(Some("high"))).expect("high");
    let b = render_fixture(FLASH_NEXT, &msgs(), None, flags(Some("xhigh"))).expect("xhigh");
    assert_eq!(a, b, "the retry must produce the native spelling's bytes");
}

/// Tools present is the shape the failure was found in (the round-trip rig
/// sends two tools), and the template puts the directive inside the tool
/// system block — a different branch from the no-tools path above.
#[test]
fn retry_also_covers_the_tools_branch() {
    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    })];
    let out = render_fixture(FLASH_NEXT, &msgs(), Some(&tools), flags(Some("high")))
        .expect("`high` with tools must render");
    assert!(out.contains(XHIGH_SENTENCE), "{out}");
    assert!(out.contains("read_file"), "tools must still be rendered:\n{out}");
}

/// Thinking off must never carry an effort directive into the render, retry
/// or not — the effort is resolved to `none` upstream and the template's
/// effort block is gated on `enable_thinking`.
#[test]
fn thinking_off_never_reaches_the_effort_block() {
    let out = render_fixture(
        FLASH_NEXT,
        &msgs(),
        None,
        RenderFlags {
            enable_thinking: false,
            reasoning_effort: Some("none"),
            ..Default::default()
        },
    )
    .expect("thinking-off renders");
    assert!(!out.contains("Reasoning effort is set to"), "{out}");
}
