// SPDX-License-Identifier: AGPL-3.0-only
//
// `MsgEntry` + the pre-loop builder that turns the inbound
// `ChatCompletionRequest.messages` into the local representation
// used by every downstream phase (json_messages, loop detector,
// task pin, observation mask, …).
//
// Lifted out of `chat::chat_completions_inner` (wave 4g) so the
// orchestrator stays under the 500-LoC cap.

use axum::http::StatusCode;
use axum::response::Response;

use atlas_core::config::VisionConfig;

use crate::ir::{ContentPart, ImageData, Message, Role};

use super::super::compact::openai_error_response;

/// Per-message data: role, content text, optional structured
/// `tool_calls`, and image-part count for the Jinja vision-marker
/// expansion. `pub(super)` so `chat::chat_completions_inner` and
/// the other `chat/*` sub-files can read every field.
pub(super) struct MsgEntry {
    pub(super) role: String,
    pub(super) content: String,
    /// Structured tool_calls for the Jinja template (arguments
    /// pre-parsed to dicts).
    pub(super) tool_calls: Option<Vec<serde_json::Value>>,
    /// Number of image content parts on this message. When > 0
    /// the json_messages builder emits a structured content array
    /// so the Jinja template can render
    /// `<|vision_start|><|image_pad|><|vision_end|>` markers.
    pub(super) image_count: usize,
    /// Historical reasoning trace from a prior assistant turn (the
    /// `<think>...</think>` body). Forwarded from `IncomingMessage`
    /// and passed to the Jinja template so the template can
    /// rehydrate the historical `<think>` block. Empty/None ⇒ no
    /// `<think>` block emitted for this message — prevents the
    /// empty-`<think></think>` poisoning pattern that triggers
    /// premature `<|im_end|>` (vLLM/SGLang #131, MLC commit d75d64e).
    pub(super) reasoning_content: Option<String>,
}

/// Outputs of [`build_msg_entries`]. Bundled as a struct because
/// the caller threads each field through five later phases.
pub(super) struct BuildOut {
    pub(super) messages: Vec<MsgEntry>,
    pub(super) cwd_hint: Option<String>,
    pub(super) image_pixels: Vec<(Vec<f32>, usize, usize)>,
    pub(super) image_pad_counts: Vec<usize>,
}

/// Append the encoder-input string for every image part on `m` to
/// `all_images`, growing `image_pad_counts` in lockstep (each pad count is
/// filled in later by the vision preprocessor). Shared by the tool-message
/// branch and the normal branch so images ride every role uniformly —
/// including tool results, the motivating case for issue #165.
fn collect_message_images(
    m: &Message,
    all_images: &mut Vec<String>,
    image_pad_counts: &mut Vec<usize>,
) {
    for part in &m.content {
        if let ContentPart::Image(src) = part {
            let uri = match &src.data {
                ImageData::Base64(s) | ImageData::Url(s) => s.clone(),
            };
            all_images.push(uri);
            image_pad_counts.push(0);
        }
    }
}

#[allow(clippy::result_large_err)]
pub(super) fn build_msg_entries(
    vision_config: Option<&VisionConfig>,
    input: &[Message],
    tools_active: bool,
) -> Result<BuildOut, Response> {
    let mut messages: Vec<MsgEntry> = Vec::with_capacity(input.len());
    let mut all_images: Vec<String> = Vec::new();
    let mut image_pad_counts: Vec<usize> = Vec::new();
    let mut consecutive_tool_errors: u32 = 0;
    // BW1 bash-wandering watchdog: tally tool-call productivity across the
    // conversation so a steering nudge can fire if the agent explores/runs
    // many commands without ever writing the deliverable (gap #9).
    let mut total_tool_calls: usize = 0;
    let mut productive_tool_calls: usize = 0;

    // F6 (2026-05-26): `last_query_index` was previously used to gate
    // an empty `<think>\n\n</think>\n\n` injection for historical
    // assistant turns. The Jinja template already does this gating
    // itself (via its own `ns.last_query_index` computation) and the
    // injection here was the source of empty-think poisoning. Removed.
    for m in input.iter() {
        let text = m.text();

        // Preserve structured tool_calls for the Jinja template.
        // Always extract from assistant messages — past turns may
        // carry tool_calls that the template MUST render even when
        // the current request didn't pass `tools`. `tc.arguments` is
        // already structured JSON in the IR (parsed at the adapter
        // boundary), so we forward it directly.
        let tool_calls_json = if m.role == Role::Assistant && !m.tool_calls.is_empty() {
            let parsed: Vec<serde_json::Value> = m
                .tool_calls
                .iter()
                .map(|tc| {
                    serde_json::json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments
                        }
                    })
                })
                .collect();
            Some(parsed)
        } else {
            None
        };

        // BW1: tally tool-call productivity (write/edit/build-run vs explore).
        if m.role == Role::Assistant && !m.tool_calls.is_empty() {
            for tc in &m.tool_calls {
                total_tool_calls += 1;
                if crate::hint_injector::tool_call_is_productive(&tc.name, &tc.arguments) {
                    productive_tool_calls += 1;
                }
            }
        }

        // Tool-response messages: pass raw content; Jinja template
        // handles `<tool_response>` wrapping and consecutive
        // grouping.
        if tools_active && m.role == Role::Tool {
            let mut text = text;
            if crate::hint_injector::looks_like_error(&text) {
                consecutive_tool_errors += 1;
                crate::hint_injector::inject_hints(&mut text, consecutive_tool_errors);
            } else {
                consecutive_tool_errors = 0;
            }
            messages.push(MsgEntry {
                role: "tool".into(),
                content: text,
                tool_calls: None,
                image_count: m.image_count(),
                reasoning_content: None,
            });
            collect_message_images(m, &mut all_images, &mut image_pad_counts);
            continue;
        }

        let image_count = m.image_count();
        // Wave 3 (2026-05-26): `ATLAS_STRIP_REASONING_HISTORY=1` drops
        // historical reasoning_content entirely. Matches MLC commit
        // d75d64e (Apr 2026) `strip_reasoning_in_history` for qwen3,
        // whose PR description matches Atlas's Wave-1 failure mode
        // verbatim: echoing prior `<think>` traces makes the next turn
        // emit `<|im_end|>` prematurely AND seeds loop-attractor drift
        // on prior-failed-attempt token patterns (the `lean://` loop
        // observed in the Wave-1 opencode probe).
        let strip_reasoning = std::env::var("ATLAS_STRIP_REASONING_HISTORY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        messages.push(MsgEntry {
            role: m.role.as_wire().to_string(),
            content: text,
            tool_calls: tool_calls_json,
            image_count,
            // F1: forward reasoning_content for assistant messages only.
            // Wave 3: when strip_reasoning=true, drop it for ALL turns,
            // forcing the template back to the pre-F1 "clean content
            // only" rendering shape — but without re-introducing the
            // empty-`<think>\n\n</think>\n\n` poisoning, because F6's
            // template change skips the wrapper when reasoning_content
            // is empty.
            reasoning_content: if m.role == Role::Assistant && !strip_reasoning {
                m.reasoning
                    .as_ref()
                    .map(|r| r.text.trim().to_string())
                    .filter(|s| !s.is_empty())
            } else {
                None
            },
        });
        collect_message_images(m, &mut all_images, &mut image_pad_counts);
    }

    // Extract working directory from the system message if present.
    let cwd_hint: Option<String> = messages.iter().find(|m| m.role == "system").and_then(|m| {
        for line in m.content.lines() {
            let lower = line.to_lowercase();
            if (lower.contains("working directory")
                || lower.contains("working_directory")
                || lower.contains("cwd:"))
                && let Some(pos) = line.find(':')
            {
                let path = line[pos + 1..]
                    .trim()
                    .trim_matches(|c| c == '`' || c == '"' || c == '\'');
                if !path.is_empty() {
                    return Some(path.to_string());
                }
            }
        }
        None
    });

    // Inject CWD hint into the system message (NOT tool definitions —
    // those go to the Jinja template).
    if tools_active && let Some(ref cwd) = cwd_hint {
        let hints = format!("\n<environment>\nworking_directory: {cwd}\n</environment>");
        if let Some(first) = messages.first_mut()
            && first.role == "system"
        {
            first.content.push_str(&hints);
        }
    }

    // Neutralize a content-free leading system message. Clients (notably
    // Open WebUI's empty RAG/context template) inject a system message
    // carrying NO instruction — e.g. `"User Context:\n\n"` (trims to the
    // bare label `User Context:`). Models react to a content-free system
    // directive by producing terse / prematurely-terminated output
    // (isolated 2026-05-17: removing it 3x'd generation length on the
    // 3D-chess prompt). We can't fix the client, so Atlas adapts: treat
    // such a message as absent so a degenerate client prompt can't poison
    // generation. Conservative — only an empty body or a single short
    // bare `Label:` line qualifies; any substantive prompt is untouched.
    if messages
        .first()
        .is_some_and(|m| m.role == "system" && is_vacuous_system_content(&m.content))
    {
        let removed = messages.remove(0);
        tracing::info!(
            dropped = %removed.content.trim(),
            "Dropped content-free client system message (would bias the model toward terse output)"
        );
    }

    // Preprocess images. One shared fail-fast point: if images were
    // supplied but the model has no vision encoder, reject the request
    // (issue #165) instead of silently dropping the user's input with a
    // 200 — the old text-only behavior lost images without any signal.
    let mut image_pixels: Vec<(Vec<f32>, usize, usize)> = Vec::new();
    if !all_images.is_empty() {
        let Some(vcfg) = vision_config else {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                "this model does not accept image input (no vision config)".to_string(),
            ));
        };
        for (idx, uri) in all_images.iter().enumerate() {
            match spark_model::vision_preprocess::preprocess_image(uri, vcfg) {
                Ok((pixels, grid_h, grid_w)) => {
                    image_pad_counts[idx] = spark_model::vision_preprocess::image_pad_count(
                        grid_h,
                        grid_w,
                        vcfg.spatial_merge_size,
                    );
                    image_pixels.push((pixels, grid_h, grid_w));
                }
                Err(e) => {
                    return Err(openai_error_response(
                        StatusCode::BAD_REQUEST,
                        format!("Image decode error: {e}"),
                    ));
                }
            }
        }
    }

    // BW1 bash-wandering watchdog: if the agent has run many tool calls with
    // no productive file output, append a steering nudge to the most recent
    // tool response (what the model reads just before its next action). Gated
    // by ATLAS_BASH_WANDER_WATCHDOG (PCND, default-off).
    if tools_active
        && let Some(hint) =
            crate::hint_injector::bash_wander_hint(total_tool_calls, productive_tool_calls)
        && let Some(last_tool) = messages.iter_mut().rev().find(|e| e.role == "tool")
    {
        last_tool.content.push_str(&hint);
    }

    Ok(BuildOut {
        messages,
        cwd_hint,
        image_pixels,
        image_pad_counts,
    })
}

/// True when a system message carries no actual instruction and should
/// be treated as absent. Conservative by design — a substantive prompt
/// must never be stripped:
///   * empty / whitespace-only body, OR
///   * a single short bare label line ending in ':' with nothing after
///     it (e.g. `User Context:`, `Context:`, `System:`) — the residue
///     of an empty client template (Open WebUI's RAG/context block).
/// Anything multi-line, or with any text past the colon, is a real
/// prompt and returns false.
fn is_vacuous_system_content(content: &str) -> bool {
    let t = content.trim();
    if t.is_empty() {
        return true;
    }
    if !t.contains('\n') && t.len() <= 32 && t.ends_with(':') {
        let label = &t[..t.len() - 1];
        return !label.is_empty()
            && label
                .chars()
                .all(|c| c.is_ascii_alphabetic() || c == ' ' || c == '_' || c == '-');
    }
    false
}

#[cfg(test)]
mod vacuous_system_tests {
    use super::is_vacuous_system_content;

    #[test]
    fn empty_or_whitespace_is_vacuous() {
        assert!(is_vacuous_system_content(""));
        assert!(is_vacuous_system_content("   \n\t  "));
    }

    #[test]
    fn open_webui_empty_context_residue_is_vacuous() {
        // The exact 2026-05-17 field artifact.
        assert!(is_vacuous_system_content("User Context:\n\n"));
        assert!(is_vacuous_system_content("Context:"));
        assert!(is_vacuous_system_content("  System:  "));
    }

    #[test]
    fn substantive_prompt_is_not_vacuous() {
        assert!(!is_vacuous_system_content(
            "User Context:\nThe user is a senior Rust engineer."
        ));
        assert!(!is_vacuous_system_content("You are a helpful assistant."));
        assert!(!is_vacuous_system_content("Always answer in French."));
        // Label-like but with a real payload after the colon.
        assert!(!is_vacuous_system_content("Role: expert chess coach"));
        // Long single line ending in ':' is unusual prose, not a bare
        // header — keep it (avoid false-strip).
        assert!(!is_vacuous_system_content(
            "Summarize the following transcript and then ask the user this:"
        ));
    }
}

#[cfg(test)]
mod build_tests {
    use super::build_msg_entries;
    use crate::ir::message::{ContentPart, ImageData, ImageSource, Message, Role};
    use axum::http::StatusCode;

    fn assert_bad_request(msgs: &[Message], tools_active: bool) {
        match build_msg_entries(None, msgs, tools_active) {
            Ok(_) => panic!("expected 400, got Ok"),
            Err(resp) => assert_eq!(resp.status(), StatusCode::BAD_REQUEST),
        }
    }

    fn text(role: Role, t: &str) -> Message {
        Message {
            role,
            content: vec![ContentPart::Text(t.into())],
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            reasoning: None,
            tool_error: false,
        }
    }

    fn image(role: Role) -> Message {
        Message {
            role,
            content: vec![
                ContentPart::Image(ImageSource {
                    data: ImageData::Base64("data:image/png;base64,AAA".into()),
                }),
                ContentPart::Text("result".into()),
            ],
            tool_calls: Vec::new(),
            tool_call_id: Some("c1".into()),
            name: None,
            reasoning: None,
            tool_error: false,
        }
    }

    #[test]
    fn text_only_builds_without_vision_config() {
        let msgs = vec![text(Role::User, "hello")];
        let out = build_msg_entries(None, &msgs, false).expect("text-only ok");
        assert_eq!(out.messages.len(), 1);
        assert_eq!(out.messages[0].image_count, 0);
    }

    #[test]
    fn user_image_without_vision_config_is_rejected() {
        // Previously: silently dropped (200). Now: fail fast.
        assert_bad_request(&[text(Role::User, "hi"), image(Role::User)], false);
    }

    #[test]
    fn tool_result_image_is_collected_and_rejected_without_vision() {
        // Proves the tool branch now COUNTS + COLLECTS images (it used to
        // hardcode image_count: 0 and `continue` before collection): the
        // fail-fast only fires when an image was actually collected.
        assert_bad_request(&[text(Role::User, "look"), image(Role::Tool)], true);
    }
}
