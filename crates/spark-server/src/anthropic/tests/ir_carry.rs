// SPDX-License-Identifier: AGPL-3.0-only
//
// Anthropic request-translation: images and reasoning are carried into
// the lowered OpenAI request instead of being dropped (issue #165).

use super::super::translate::anthropic_to_chat_request_json;
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
