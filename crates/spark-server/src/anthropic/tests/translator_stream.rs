// SPDX-License-Identifier: AGPL-3.0-only
//
// Golden tests for the Anthropic streaming translator's event framing.
// These characterize the current behavior so the parse/typing refactor
// (issue #165 response IR) can't silently change the wire output that
// Claude Code depends on.

use super::super::translator::{AnthropicTranslator, SseEvent};

fn drive(chunks: &[serde_json::Value]) -> Vec<SseEvent> {
    let mut t = AnthropicTranslator::new("m".to_string());
    let mut out = Vec::new();
    for c in chunks {
        t.process_openai_chunk(c, &mut out);
    }
    t.finalize(&mut out);
    out
}

fn names(evs: &[SseEvent]) -> Vec<&str> {
    evs.iter().map(|e| e.event.as_str()).collect()
}

#[test]
fn text_stream_framing() {
    let evs = drive(&[
        serde_json::json!({"id":"chatcmpl-x","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}),
        serde_json::json!({"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}),
        serde_json::json!({"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}),
    ]);
    assert_eq!(
        names(&evs),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(evs[1].data["content_block"]["type"], "text");
    assert_eq!(evs[2].data["delta"]["type"], "text_delta");
    assert_eq!(evs[2].data["delta"]["text"], "Hi");
    assert_eq!(evs[4].data["delta"]["stop_reason"], "end_turn");
    assert_eq!(evs[4].data["usage"]["output_tokens"], 1);
}

#[test]
fn thinking_then_text_framing() {
    let evs = drive(&[
        serde_json::json!({"id":"chatcmpl-x","choices":[{"delta":{"reasoning_content":"think"},"finish_reason":null}]}),
        serde_json::json!({"choices":[{"delta":{"content":"answer"},"finish_reason":null}]}),
        serde_json::json!({"choices":[{"delta":{},"finish_reason":"stop"}]}),
    ]);
    assert_eq!(
        names(&evs),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(evs[1].data["content_block"]["type"], "thinking");
    assert_eq!(evs[2].data["delta"]["type"], "thinking_delta");
    assert_eq!(evs[2].data["delta"]["thinking"], "think");
    assert_eq!(evs[5].data["delta"]["type"], "text_delta");
    assert_eq!(evs[5].data["delta"]["text"], "answer");
}

#[test]
fn tool_call_stream_framing() {
    let evs = drive(&[
        serde_json::json!({"id":"chatcmpl-x","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}),
        serde_json::json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}),
        serde_json::json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\"SF\"}"}}]},"finish_reason":null}]}),
        serde_json::json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
    ]);
    assert_eq!(
        names(&evs),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(evs[1].data["content_block"]["type"], "tool_use");
    assert_eq!(evs[1].data["content_block"]["name"], "get_weather");
    assert_eq!(evs[1].data["content_block"]["id"], "call_1");
    assert_eq!(evs[2].data["delta"]["type"], "input_json_delta");
    assert_eq!(evs[2].data["delta"]["partial_json"], "{\"city\":\"SF\"}");
    assert_eq!(evs[4].data["delta"]["stop_reason"], "tool_use");
}
