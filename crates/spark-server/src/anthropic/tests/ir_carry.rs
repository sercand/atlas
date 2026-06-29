// SPDX-License-Identifier: AGPL-3.0-only
//
// Anthropic request-translation: images and reasoning are carried into
// the lowered OpenAI request instead of being dropped (issue #165).

use super::super::translate::{anthropic_to_chat_request_json, chat_to_anthropic_response};
use super::super::types::MessagesRequest;
use crate::openai::ChatCompletionRequest;

/// Lower an Anthropic request body and re-parse it as a
/// `ChatCompletionRequest` (the same path `handlers::messages` takes), so
/// assertions can read `ParsedContent` directly.
fn lower(req_json: serde_json::Value) -> ChatCompletionRequest {
    let req: MessagesRequest = serde_json::from_value(req_json).expect("MessagesRequest");
    let chat_json = anthropic_to_chat_request_json(&req);
    serde_json::from_value(chat_json).expect("ChatCompletionRequest")
}

#[test]
fn user_image_block_is_carried() {
    let chat = lower(serde_json::json!({
        "model": "m", "max_tokens": 16,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}}
        ]}]
    }));
    let user = chat
        .messages
        .iter()
        .find(|m| m.role == "user")
        .expect("user msg");
    assert_eq!(
        user.content.images,
        vec!["data:image/png;base64,AAA".to_string()]
    );
    assert_eq!(user.content.text, "what is this?");
}

#[test]
fn tool_result_image_block_is_carried() {
    let chat = lower(serde_json::json!({
        "model": "m", "max_tokens": 16,
        "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "text", "text": "see screenshot"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "BBB"}}
            ]}
        ]}]
    }));
    let tool = chat
        .messages
        .iter()
        .find(|m| m.role == "tool")
        .expect("tool msg");
    assert_eq!(
        tool.content.images,
        vec!["data:image/jpeg;base64,BBB".to_string()]
    );
    assert_eq!(tool.content.text, "see screenshot");
}

#[test]
fn thinking_block_becomes_reasoning_content() {
    let chat = lower(serde_json::json!({
        "model": "m", "max_tokens": 16,
        "messages": [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "let me reason"},
            {"type": "text", "text": "the answer is 4"}
        ]}]
    }));
    let asst = chat
        .messages
        .iter()
        .find(|m| m.role == "assistant")
        .expect("assistant msg");
    assert_eq!(asst.reasoning_content.as_deref(), Some("let me reason"));
    assert_eq!(asst.content.text, "the answer is 4");
}

#[test]
fn url_image_source_is_carried() {
    let chat = lower(serde_json::json!({
        "model": "m", "max_tokens": 16,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}
        ]}]
    }));
    let user = chat
        .messages
        .iter()
        .find(|m| m.role == "user")
        .expect("user msg");
    assert_eq!(
        user.content.images,
        vec!["https://example.com/a.png".to_string()]
    );
}

// ── response direction (chat_to_anthropic_response) ──

#[test]
fn response_maps_reasoning_text_tools_and_usage() {
    let v = serde_json::json!({
        "id": "chatcmpl-abc",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "choices": [{
            "message": {
                "reasoning_content": "thinking",
                "content": "hello",
                "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{\"x\":1}"}}]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let json = serde_json::to_value(chat_to_anthropic_response(&v, "m".into())).unwrap();
    assert_eq!(json["id"], "msg_abc");
    assert_eq!(json["model"], "m");
    assert_eq!(json["stop_reason"], "tool_use");
    assert_eq!(json["usage"]["input_tokens"], 10);
    assert_eq!(json["usage"]["output_tokens"], 5);
    assert_eq!(json["content"][0]["type"], "thinking");
    assert_eq!(json["content"][0]["thinking"], "thinking");
    assert_eq!(json["content"][1]["type"], "text");
    assert_eq!(json["content"][1]["text"], "hello");
    assert_eq!(json["content"][2]["type"], "tool_use");
    assert_eq!(json["content"][2]["name"], "f");
    assert_eq!(json["content"][2]["input"], serde_json::json!({"x": 1}));
}

#[test]
fn response_missing_id_becomes_msg_unknown() {
    let v = serde_json::json!({
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]
    });
    let json = serde_json::to_value(chat_to_anthropic_response(&v, "m".into())).unwrap();
    assert_eq!(json["id"], "msg_unknown");
    assert_eq!(json["stop_reason"], "end_turn");
    assert_eq!(json["content"][0]["text"], "hi");
}

#[test]
fn response_empty_content_omits_text_block() {
    let v = serde_json::json!({
        "id": "chatcmpl-x",
        "choices": [{"message": {"content": ""}, "finish_reason": "stop"}]
    });
    let json = serde_json::to_value(chat_to_anthropic_response(&v, "m".into())).unwrap();
    assert_eq!(json["content"].as_array().unwrap().len(), 0);
}

#[test]
fn text_only_user_keeps_plain_string_content() {
    // No images → content stays a plain string (behavior-preserving for
    // the overwhelmingly common case).
    let req: MessagesRequest = serde_json::from_value(serde_json::json!({
        "model": "m", "max_tokens": 16,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    }))
    .unwrap();
    let chat_json = anthropic_to_chat_request_json(&req);
    assert_eq!(chat_json["messages"][0]["content"], serde_json::json!("hi"));
}
