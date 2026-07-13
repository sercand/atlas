// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn url_of(a: &Annotation) -> (usize, usize, &str, &str) {
    match a {
        Annotation::UrlCitation {
            url_citation:
                UrlCitation {
                    start_index,
                    end_index,
                    url,
                    title,
                },
        } => (*start_index, *end_index, url.as_str(), title.as_str()),
    }
}

#[test]
fn bare_url_extracted() {
    let got = extract_url_annotations("see https://example.com/foo for more").unwrap();
    assert_eq!(got.len(), 1);
    let (s, e, u, t) = url_of(&got[0]);
    assert_eq!(u, "https://example.com/foo");
    assert_eq!(t, "https://example.com/foo");
    assert_eq!(s, 4);
    assert_eq!(e, 4 + "https://example.com/foo".len());
}

#[test]
fn trailing_sentence_punct_stripped() {
    let got = extract_url_annotations("go to https://example.com.").unwrap();
    let (_, _, u, _) = url_of(&got[0]);
    assert_eq!(u, "https://example.com");
}

#[test]
fn wikipedia_parens_preserved() {
    let got = extract_url_annotations("see https://en.wikipedia.org/wiki/Foo_(bar) now").unwrap();
    let (_, _, u, _) = url_of(&got[0]);
    assert_eq!(u, "https://en.wikipedia.org/wiki/Foo_(bar)");
}

#[test]
fn markdown_link_uses_title() {
    let got = extract_url_annotations("read [the docs](https://example.com/api) today").unwrap();
    assert_eq!(got.len(), 1);
    let (_, _, u, t) = url_of(&got[0]);
    assert_eq!(u, "https://example.com/api");
    assert_eq!(t, "the docs");
}

#[test]
fn url_in_fenced_code_skipped() {
    let input = "run this:\n```bash\ncurl https://example.com\n```\ndone";
    assert!(extract_url_annotations(input).is_none());
}

#[test]
fn url_in_inline_code_skipped() {
    let input = "use `curl https://example.com` to fetch";
    assert!(extract_url_annotations(input).is_none());
}

#[test]
fn multiple_urls_sorted_by_position() {
    let input = "first https://a.example.com and [second](https://b.example.com)";
    let got = extract_url_annotations(input).unwrap();
    assert_eq!(got.len(), 2);
    let (s0, _, u0, _) = url_of(&got[0]);
    let (s1, _, u1, _) = url_of(&got[1]);
    assert!(s0 < s1);
    assert_eq!(u0, "https://a.example.com");
    assert_eq!(u1, "https://b.example.com");
}

#[test]
fn non_http_ignored() {
    assert!(extract_url_annotations("ftp://example.com not a citation").is_none());
}

#[test]
fn empty_input_returns_none() {
    assert!(extract_url_annotations("").is_none());
    assert!(extract_url_annotations("no URLs here").is_none());
}

#[test]
fn query_and_fragment_preserved() {
    let got = extract_url_annotations("see https://example.com/p?q=1&r=2#frag here").unwrap();
    let (_, _, u, _) = url_of(&got[0]);
    assert_eq!(u, "https://example.com/p?q=1&r=2#frag");
}

// TODO: `mask_code_spans` was an internal helper that no longer exists
// after the URL-annotations refactor. The remaining call to
// `extract_url_annotations` is exercised by the other tests in this file;
// re-add a UTF-8 boundary test once the new internal mask helper has a
// stable name.

fn lower_with_tools(
    tools: serde_json::Value,
) -> Result<ChatCompletionRequest, LowerResponsesError> {
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "input": "ping",
        "tools": tools,
    }))
    .expect("ResponsesRequest deserialize");
    lower_responses_to_chat(req, |_| None)
}

#[test]
fn responses_flat_function_tool_accepted() {
    // OpenAI's official Python SDK sends function tools in the flat
    // shape `{type, name, description, parameters}` — no nested
    // `function` object. Atlas must accept both shapes.
    let chat = lower_with_tools(serde_json::json!([
        {
            "type": "function",
            "name": "get_weather",
            "description": "look up weather",
            "parameters": {"type": "object", "properties": {"loc": {"type": "string"}}, "required": ["loc"]}
        }
    ])).expect("flat-form function tool should parse");
    let tools = chat.tools.expect("tools present");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].tool_type, "function");
    assert_eq!(tools[0].function.name, "get_weather");
}

#[test]
fn responses_nested_function_tool_still_accepted() {
    // Backwards-compat: chat-completions-style `{type, function:{...}}`
    // must keep working since older clients send it.
    let chat = lower_with_tools(serde_json::json!([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object"}
            }
        }
    ]))
    .expect("nested-form function tool should parse");
    let tools = chat.tools.expect("tools present");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");
}

#[test]
fn responses_flat_tool_choice_accepted() {
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "input": "ping",
        "tool_choice": {"type": "function", "name": "get_weather"},
    }))
    .unwrap();
    let chat = lower_responses_to_chat(req, |_| None).expect("flat tool_choice");
    match chat.tool_choice {
        Some(crate::tool_parser::ToolChoice::Specific { function }) => {
            assert_eq!(function.name, "get_weather");
        }
        other => panic!("expected Specific tool_choice, got {other:?}"),
    }
}

#[test]
fn responses_string_tool_choice_accepted() {
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "input": "ping",
        "tool_choice": "required",
    }))
    .unwrap();
    let chat = lower_responses_to_chat(req, |_| None).expect("string tool_choice");
    match chat.tool_choice {
        Some(crate::tool_parser::ToolChoice::Mode(s)) => {
            assert_eq!(s, "required");
        }
        other => panic!("expected Mode tool_choice, got {other:?}"),
    }
}

#[test]
fn markdown_link_with_parens_in_url_preserved() {
    // Wikipedia URLs contain `(...)` which the bare `find(')')` would
    // truncate. Verify the balanced-paren scan keeps the full URL.
    let got =
        extract_url_annotations("see [Foo (bar)](https://en.wikipedia.org/wiki/Foo_(bar)) here")
            .unwrap();
    assert_eq!(got.len(), 1);
    let (_, _, u, t) = url_of(&got[0]);
    assert_eq!(u, "https://en.wikipedia.org/wiki/Foo_(bar)");
    assert_eq!(t, "Foo (bar)");
}

#[test]
fn responses_input_image_string_url_is_carried() {
    // Regression: the Responses adapter used to drop images entirely
    // (`images: Vec::new()`), silently losing multimodal input.
    let item = serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": "what is this?"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAA"}
        ]
    });
    let m = IncomingMessage::from_responses_input_item(&item).expect("message");
    assert_eq!(m.content.text, "what is this?");
    assert_eq!(
        m.content.images,
        vec!["data:image/png;base64,AAA".to_string()]
    );
}

#[test]
fn responses_input_image_object_url_is_carried() {
    // Some SDKs nest the url: `{image_url: {url: "..."}}`.
    let item = serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": {"url": "https://example.com/a.png"}}
        ]
    });
    let m = IncomingMessage::from_responses_input_item(&item).expect("message");
    assert_eq!(
        m.content.images,
        vec!["https://example.com/a.png".to_string()]
    );
    assert_eq!(m.content.text, "");
}

#[test]
fn responses_function_call_output_image_is_carried() {
    // #165 parity for the Responses surface: a screenshot returned by a
    // tool (`function_call_output` with structured output parts) must
    // reach the vision encoder, exactly like Anthropic tool_result
    // images and chat-completions role:"tool" array content.
    let item = serde_json::json!({
        "type": "function_call_output",
        "call_id": "call_1",
        "output": [
            {"type": "output_text", "text": "screenshot follows"},
            {"type": "input_image", "image_url": "data:image/png;base64,BBB"}
        ]
    });
    let m = IncomingMessage::from_responses_input_item(&item).expect("tool message");
    assert_eq!(m.role, "tool");
    assert_eq!(m.tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(m.content.text, "screenshot follows");
    assert_eq!(
        m.content.images,
        vec!["data:image/png;base64,BBB".to_string()]
    );
}

#[test]
fn responses_function_call_output_string_unchanged() {
    let item = serde_json::json!({
        "type": "function_call_output",
        "call_id": "call_2",
        "output": "plain result"
    });
    let m = IncomingMessage::from_responses_input_item(&item).expect("tool message");
    assert_eq!(m.content.text, "plain result");
    assert!(m.content.images.is_empty());
}

#[test]
fn responses_function_call_output_opaque_array_stringified() {
    // Out-of-spec array (no recognizable parts) keeps the historical
    // stringified-JSON behavior instead of silently emptying the result.
    let opaque = serde_json::json!([{"weather": "sunny", "temp_c": 21}]);
    let item = serde_json::json!({
        "type": "function_call_output",
        "call_id": "call_3",
        "output": opaque.clone()
    });
    let m = IncomingMessage::from_responses_input_item(&item).expect("tool message");
    assert_eq!(m.content.text, opaque.to_string());
    assert!(m.content.images.is_empty());
}

#[test]
fn responses_in_progress_event_name() {
    let ev = ResponsesStreamEvent::InProgress {
        sequence_number: 1,
        response: ResponsesStreamEnvelope {
            id: "resp_test".into(),
            object: "response",
            created_at: 0,
            model: "m".into(),
            status: "in_progress",
            metadata: None,
        },
    };
    assert_eq!(responses_event_name(&ev), "response.in_progress");
}

// ── return_token_ids wire format ────────────────────────────────────

#[test]
fn token_ids_absent_by_default_keeps_wire_byte_identical() {
    // PCND: a client that did not opt in must see no `token_ids` key.
    let chunk = ChatCompletionChunk::content_chunk("m", "id", "hi".into());
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(!json.contains("token_ids"), "default wire changed: {json}");
    // Empty `with_token_ids` is a no-op (still absent).
    let chunk =
        ChatCompletionChunk::content_chunk("m", "id", "hi".into()).with_token_ids(Vec::new());
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(!json.contains("token_ids"));
}

#[test]
fn with_token_ids_stamps_first_choice() {
    let chunk =
        ChatCompletionChunk::content_chunk("m", "id", "hi".into()).with_token_ids(vec![10, 20, 30]);
    assert_eq!(chunk.choices[0].token_ids, vec![10, 20, 30]);
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(json.contains("\"token_ids\":[10,20,30]"), "{json}");
    // No choices (usage-only chunk) → no panic, no-op.
    let usage = Usage {
        prompt_tokens: 1,
        completion_tokens: 1,
        total_tokens: 2,
        prompt_tokens_details: None,
        completion_tokens_details: None,
        time_to_first_token_ms: 0.0,
        response_tokens_per_second: 0.0,
    };
    let chunk = ChatCompletionChunk::usage_only_chunk("m", "id", usage).with_token_ids(vec![1, 2]);
    assert!(chunk.choices.is_empty());
}

// ── reasoning wire format: exactly one field ────────────────────────
// A response carrying BOTH `reasoning_content` and a `reasoning` mirror is
// rejected by strict OpenAI-compatible clients (they assert exactly one).
// Atlas emits only `reasoning_content` — these lock that contract in.

#[test]
fn reasoning_delta_emits_only_reasoning_content() {
    let chunk = ChatCompletionChunk::reasoning_chunk("m", "id", "thinking".into());
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(
        json.contains("\"reasoning_content\":\"thinking\""),
        "reasoning_content missing: {json}"
    );
    assert!(
        !json.contains("\"reasoning\":"),
        "mirror `reasoning` field leaked into stream delta: {json}"
    );
}

#[test]
fn blocking_message_emits_only_reasoning_content() {
    let msg = ChatMessage {
        role: "assistant".into(),
        reasoning_content: Some("thinking".into()),
        content: Some("hi".into()),
        tool_calls: None,
        annotations: None,
        refusal: None,
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(
        json.contains("\"reasoning_content\":\"thinking\""),
        "reasoning_content missing: {json}"
    );
    assert!(
        !json.contains("\"reasoning\":"),
        "mirror `reasoning` field leaked into message: {json}"
    );
}

// ── Thinking directive (client channels → ir::ThinkingDirective) ──

use crate::ir::ThinkingDirective;

fn chat_req(body: serde_json::Value) -> ChatCompletionRequest {
    serde_json::from_value(body).expect("valid chat request")
}

fn base_body() -> serde_json::Value {
    serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
    })
}

#[test]
fn silent_request_is_unspecified() {
    let req = chat_req(base_body());
    assert_eq!(
        req.client_thinking_directive(),
        ThinkingDirective::Unspecified
    );
    assert!(!req.client_thinking_directive().is_explicit());
}

#[test]
fn anthropic_thinking_channel() {
    // type=disabled wins even with a budget present.
    let mut b = base_body();
    b["thinking"] = serde_json::json!({"type": "disabled", "budget_tokens": 100});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );

    let mut b = base_body();
    b["thinking"] = serde_json::json!({"type": "enabled", "budget_tokens": 512});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: Some(512) }
    );

    // Adaptive / budget-less thinking object → think as long as needed
    // (budget defers to the per-model max_thinking_budget).
    let mut b = base_body();
    b["thinking"] = serde_json::json!({"type": "adaptive"});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: None }
    );
}

#[test]
fn thinking_token_budget_channel() {
    let mut b = base_body();
    b["thinking_token_budget"] = serde_json::json!(512);
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: Some(512) }
    );

    let mut b = base_body();
    b["thinking_token_budget"] = serde_json::json!(0);
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );
}

#[test]
fn reasoning_effort_channel() {
    for (effort, expect) in [
        ("none", ThinkingDirective::Off),
        ("minimal", ThinkingDirective::On { budget: Some(64) }),
        ("low", ThinkingDirective::On { budget: Some(128) }),
        ("medium", ThinkingDirective::On { budget: Some(256) }),
        ("high", ThinkingDirective::On { budget: Some(512) }),
        ("xhigh", ThinkingDirective::On { budget: Some(1024) }),
        ("max", ThinkingDirective::On { budget: Some(1024) }),
        // Unknown efforts fall back to the conservative default budget.
        ("bogus", ThinkingDirective::On { budget: Some(256) }),
    ] {
        let mut b = base_body();
        b["reasoning"] = serde_json::json!({"effort": effort});
        assert_eq!(
            chat_req(b).client_thinking_directive(),
            expect,
            "effort={effort}"
        );
    }
}

#[test]
fn chat_template_kwargs_channel() {
    // Struct still parses as a request-body wire field.
    let kw: ChatTemplateKwargs =
        serde_json::from_str(r#"{"enable_thinking":true,"thinking_budget":1024}"#)
            .expect("should parse");
    assert_eq!(kw.enable_thinking, Some(true));
    assert_eq!(kw.thinking_budget, Some(1024));

    // Budget rung wins over the enable flag.
    let mut b = base_body();
    b["chat_template_kwargs"] =
        serde_json::json!({"enable_thinking": false, "thinking_budget": 1024});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: Some(1024) }
    );

    let mut b = base_body();
    b["chat_template_kwargs"] = serde_json::json!({"thinking_budget": 0});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );

    // enable_thinking with no explicit budget defers to the per-model
    // max_thinking_budget (budget: None), not the conservative 256-token
    // default — a hard cut force-injects </think> mid-reasoning and
    // wrecks agentic tool selection.
    let mut b = base_body();
    b["chat_template_kwargs"] = serde_json::json!({"enable_thinking": true});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: None }
    );

    let mut b = base_body();
    b["chat_template_kwargs"] = serde_json::json!({"enable_thinking": false});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Off
    );

    // Empty kwargs object carries no intent.
    let mut b = base_body();
    b["chat_template_kwargs"] = serde_json::json!({});
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Unspecified
    );
}

#[test]
fn legacy_enable_thinking_channel() {
    let mut b = base_body();
    b["enable_thinking"] = serde_json::json!(true);
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::On { budget: None }
    );

    // false is the serde default — indistinguishable from absent, so it
    // must NOT count as an explicit opt-out.
    let mut b = base_body();
    b["enable_thinking"] = serde_json::json!(false);
    assert_eq!(
        chat_req(b).client_thinking_directive(),
        ThinkingDirective::Unspecified
    );
}

// ── Legacy /v1/completions echo + logprobs wire types ──

#[test]
fn completion_request_echo_logprobs_n_deser() {
    let req: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "prompt": "hello",
        "echo": true,
        "logprobs": 3,
        "n": 2,
        "max_tokens": 0,
        "stream_options": {"include_usage": true},
        "user": "eval-harness",
        "suffix": "tail",
        "best_of": 2,
    }))
    .expect("valid completion request");
    assert!(req.echo);
    assert_eq!(req.logprobs, Some(3));
    assert_eq!(req.n, 2);
    assert_eq!(req.max_tokens, 0);
    assert!(req.stream_options.expect("stream_options").include_usage);
}

#[test]
fn completion_request_defaults_match_openai_spec() {
    // echo=false, n=1, logprobs absent — the spec defaults; a request
    // that names none of the new fields must behave exactly as before.
    let req: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "prompt": "hello",
    }))
    .expect("valid completion request");
    assert!(!req.echo);
    assert_eq!(req.n, 1);
    assert!(req.logprobs.is_none());
    assert!(req.stream_options.is_none());
}

#[test]
fn completion_logprobs_serializes_four_parallel_arrays_with_nulls() {
    let lp = CompletionLogprobs {
        tokens: vec!["He".into(), "llo".into()],
        token_logprobs: vec![None, Some(-0.5)],
        top_logprobs: vec![
            None,
            Some(std::collections::HashMap::from([(
                "llo".to_string(),
                -0.5f32,
            )])),
        ],
        text_offset: vec![0, 2],
    };
    let v = serde_json::to_value(&lp).expect("serialize");
    // Legacy shape: 4 parallel arrays, JSON null for the first echoed token.
    assert!(v["token_logprobs"][0].is_null());
    assert!(v["top_logprobs"][0].is_null());
    assert_eq!(v["tokens"].as_array().map(Vec::len), Some(2));
    assert_eq!(v["text_offset"][1], 2);
}

#[test]
fn completion_response_carries_system_fingerprint_and_optional_logprobs() {
    let usage = Usage {
        prompt_tokens: 1,
        completion_tokens: 1,
        total_tokens: 2,
        prompt_tokens_details: None,
        completion_tokens_details: None,
        time_to_first_token_ms: 0.0,
        response_tokens_per_second: 0.0,
    };
    let resp = CompletionResponse::new("m", "hi".into(), usage, "stop");
    let v = serde_json::to_value(&resp).expect("serialize");
    assert_eq!(v["system_fingerprint"], "fp_atlas");
    // logprobs must be ABSENT (not null) when not requested — some
    // clients treat an explicit null as a malformed logprobs block.
    assert!(v["choices"][0].get("logprobs").is_none());
}

#[test]
fn completion_request_n_bounds_are_handler_enforced() {
    // serde accepts any usize; the HANDLER rejects n==0 and n>128 with a
    // 400 (OpenAI spec bound). This test locks the parse side: values
    // arrive intact for the handler check (no silent serde clamping).
    let req: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "m", "prompt": "p", "n": 4096,
    }))
    .expect("parse");
    assert_eq!(req.n, 4096);
}
