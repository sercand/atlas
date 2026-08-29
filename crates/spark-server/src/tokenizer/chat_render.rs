// SPDX-License-Identifier: AGPL-3.0-only

//! Core Jinja chat rendering shared by every `ChatTokenizer` apply path.
//!
//! Free function on purpose (SSOT): the golden-render tests in
//! `tokenizer/tests/` call [`render_chat`] against fixture templates, so the
//! exact context-construction code that produces production prompt bytes is
//! what the tests prove — not a re-implementation that can drift.

use anyhow::{Context, Result};

use super::chat_impl::preprocess_for_render;

/// Render-time flags for [`render_chat`]. Grouped so the apply-path
/// signatures stay readable as template knobs accumulate.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RenderFlags<'a> {
    pub enable_thinking: bool,
    pub disable_tool_steering: bool,
    /// Explicit client/server reasoning-effort string. `None` falls back
    /// to the cross-template convention: `"medium"` when thinking is on
    /// (the NEUTRAL tier — no steering sentence, budget = the model's
    /// `max_thinking_budget` rung), `"none"` when off. Every template
    /// maps/validates from there — Qwen3.8 accepts `medium` verbatim and
    /// injects no directive; Mistral has no medium tier, so its
    /// Atlas-owned override maps `medium`→`high` (its standard thinking
    /// mode) and only ever sees `none` when thinking is off; Qwen3.5/3.6
    /// ignore the variable entirely. Until 2026-08-15 the thinking-on
    /// fallback was `"high"`, which Qwen3.8's template remaps to `xhigh`
    /// — every effort-silent client silently bought the MOST expensive
    /// directive tier.
    pub reasoning_effort: Option<&'a str>,
    /// Tri-state `preserve_thinking` (Qwen3.6+ dense templates). `None` =
    /// the variable is left UNDEFINED in the Jinja context so the model
    /// template's own default applies (Qwen3.6 strips historical `<think>`
    /// blocks unless true; Qwen3.8 keeps them unless explicitly false).
    /// `Some(_)` pins it. Never pass `None` as Jinja `none` — Qwen3.8's
    /// `preserve_thinking is undefined` test distinguishes the two and
    /// `none` would silently flip its default from keep to strip.
    pub preserve_thinking: Option<bool>,
    /// Diagnostic "continue final message" mode: when true AND the last
    /// message is an assistant turn, render without a generation prompt and
    /// strip the trailing `<|im_end|>` so the assistant content becomes the
    /// final prefill token(s). The OpenAI-variant path pins this false.
    pub allow_continue_final: bool,
}

/// Apply Atlas preprocessing and render the chat template to a string.
///
/// This is the single place production prompt bytes are produced for
/// Jinja-encoded models; both `apply_chat_template_jinja_with_effort` and
/// `apply_chat_template_openai_with_effort` delegate here.
pub(crate) fn render_chat(
    env: &minijinja::Environment<'static>,
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    flags: RenderFlags<'_>,
) -> Result<String> {
    let tmpl = env
        .get_template("chat")
        .context("Failed to get compiled template")?;

    // Atlas cross-cutting preprocessing (F76 arg-parse + autoclose-think
    // + think-control), applied to the model's OWN template so the
    // per-model jinja overrides that used to encode these are no longer
    // required. Inline `<|think_on|>`/`<|think_off|>` tokens, when
    // present, override the caller's `enable_thinking`.
    let (messages_for_render, enable_thinking) =
        preprocess_for_render(messages, flags.enable_thinking);
    let messages_val = minijinja::Value::from_serialize(&messages_for_render);
    let tools_val = tools.map(minijinja::Value::from_serialize);

    // Diagnostic "continue final message" mode (standard convention): see
    // [`RenderFlags::allow_continue_final`].
    let continue_final = flags.allow_continue_final
        && messages
            .last()
            .and_then(|m| m.get("role"))
            .and_then(|r| r.as_str())
            == Some("assistant");

    // Pass enable_thinking as-is to the template. The Qwen3.5 template uses it
    // to emit <think>\n (thinking) or <think>\n\n</think>\n\n (no thinking).
    // Mistral template uses reasoning_effort instead.
    // The api.rs layer controls enable_thinking based on thinking_in_tools MODEL.toml.
    // Mistral's template defaults `reasoning_effort` to "high" when
    // undefined, so we must explicitly pass "none" to disable thinking.
    //
    // Unset + thinking on ⇒ "medium": the neutral tier — no steering
    // sentence on Qwen3.8, budget = the model's own max_thinking_budget
    // rung — so a client that never heard of reasoning_effort gets the
    // model's natural behavior, not the most expensive directive. This is
    // ALSO why Qwen3.8's in-template `default('xhigh')` can never fire
    // from Atlas: we always pass an explicit string, never UNDEFINED, so
    // the template's own (most-expensive) default is unreachable and the
    // effective unset default lives in exactly one place — here.
    // Operators override per-serve via
    // `--default-chat-template-kwargs '{"reasoning_effort":"..."}'`
    // (resolved in api/chat/prepare.rs before this fallback).
    let reasoning_effort: minijinja::Value = if let Some(effort) = flags.reasoning_effort {
        effort.into()
    } else if enable_thinking {
        "medium".into()
    } else {
        "none".into()
    };
    // Tri-state → UNDEFINED (not `none`!) when unset; see RenderFlags doc.
    let preserve_thinking = flags
        .preserve_thinking
        .map(minijinja::Value::from)
        .unwrap_or(minijinja::Value::UNDEFINED);
    // Templates disagree on the reasoning-effort vocabulary, and the
    // disagreement is not knowable from the model id:
    //
    //   Qwen3.8-27B      remaps 'high' -> 'xhigh' itself, then validates
    //                    against ('xhigh','medium','low')
    //   Qwen3.8-Flash-Next  SAME validation, NO remap — `high` raises
    //   Mistral          validates against ('none','high') — `xhigh` raises
    //
    // So `reasoning_effort: "high"`, an ordinary OpenAI value, 400'd on
    // Flash-Next while working on the 27B (2026-08-29). A static table
    // cannot satisfy both directions, and a template cannot be asked what
    // it accepts. Render, and if the template raises specifically about the
    // effort value, retry ONCE with the neighbouring spelling. Costs
    // nothing on the success path and is self-correcting for a template
    // Atlas has never seen.
    let render_with = |effort: minijinja::Value| {
        tmpl.render(minijinja::context! {
            messages => messages_val.clone(),
            tools => tools_val.clone().unwrap_or(minijinja::Value::UNDEFINED),
            add_generation_prompt => !continue_final,
            enable_thinking => enable_thinking,
            reasoning_effort => effort,
            preserve_thinking => preserve_thinking.clone(),
            disable_tool_steering => flags.disable_tool_steering,
            add_vision_id => false,
        })
    };
    let mut rendered = match render_with(reasoning_effort.clone()) {
        Ok(r) => r,
        Err(e) if is_effort_rejection(&e) => {
            let asked = reasoning_effort.as_str().unwrap_or_default().to_string();
            let Some(alt) = effort_synonym(&asked) else {
                tracing::error!("Jinja template error: {e:#}");
                return Err(anyhow::anyhow!("Failed to render Jinja chat template: {e}"));
            };
            tracing::warn!(
                "chat template rejected reasoning_effort {asked:?}; retrying as {alt:?} \
                 (this template does not carry the {asked:?} remap)"
            );
            render_with(minijinja::Value::from(alt)).map_err(|e2| {
                tracing::error!("Jinja template error after effort retry: {e2:#}");
                anyhow::anyhow!("Failed to render Jinja chat template: {e2}")
            })?
        }
        Err(e) => {
            tracing::error!("Jinja template error: {e:#}");
            return Err(anyhow::anyhow!("Failed to render Jinja chat template: {e}"));
        }
    };

    if continue_final {
        // Strip the trailing end-of-turn so the assistant content is the
        // last prefill token (qwen-style templates close with
        // `<|im_end|>\n`). Trim trailing whitespace first, then the marker.
        let trimmed = rendered.trim_end();
        let stripped = trimmed.strip_suffix("<|im_end|>").unwrap_or(trimmed);
        rendered = stripped.to_string();
        tracing::info!("continue_final_message: stripped trailing EOT for prefill A/B");
    }

    crate::debug_io::dump_prompt(&rendered);
    Ok(rendered)
}

/// True when a Jinja render failed because the template validates
/// `reasoning_effort` against a fixed tuple and ours was not in it. Matched
/// on the message because that is all `raise_exception` gives back; every
/// template in the wild phrases it as "reasoning effort".
fn is_effort_rejection(e: &minijinja::Error) -> bool {
    let mut msg = e.to_string().to_ascii_lowercase();
    let mut src: Option<&dyn std::error::Error> = std::error::Error::source(e);
    while let Some(s) = src {
        msg.push(' ');
        msg.push_str(&s.to_string().to_ascii_lowercase());
        src = std::error::Error::source(s);
    }
    msg.contains("reasoning effort") || msg.contains("reasoning_effort")
}

/// The other spelling of a tier, for the one-shot retry. Only the pairs that
/// templates actually disagree about — `high` and `xhigh` are the same rung
/// under two names, and nothing else is remapped, so a genuinely unsupported
/// value still fails loudly instead of being silently downgraded.
fn effort_synonym(asked: &str) -> Option<&'static str> {
    match asked {
        "high" => Some("xhigh"),
        "xhigh" => Some("high"),
        _ => None,
    }
}
