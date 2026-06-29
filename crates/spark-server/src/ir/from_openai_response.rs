// SPDX-License-Identifier: AGPL-3.0-only
//
// Adapter: OpenAI Chat Completions response body → canonical
// `ir::ChatResponse`. A typed (`Deserialize`) view of the response
// replaces the ad-hoc `serde_json::Value::get(...)` poking the Anthropic
// surface used to do (issue #165 response direction).

use serde::Deserialize;

use super::message::ToolCall;
use super::response::{ChatResponse, FinishReason, Usage};

/// Typed view of the OpenAI response body, limited to the fields the
/// response adapters read.
#[derive(Debug, Deserialize, Default)]
pub struct OpenAiResponseView {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub usage: UsageView,
    #[serde(default)]
    pub choices: Vec<ChoiceView>,
}

#[derive(Debug, Deserialize, Default)]
pub struct UsageView {
    #[serde(default)]
    pub prompt_tokens: usize,
    #[serde(default)]
    pub completion_tokens: usize,
}

#[derive(Debug, Deserialize, Default)]
pub struct ChoiceView {
    #[serde(default)]
    pub message: MessageView,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct MessageView {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallView>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ToolCallView {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub function: FunctionView,
}

#[derive(Debug, Deserialize, Default)]
pub struct FunctionView {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

impl OpenAiResponseView {
    /// Parse a response body into the typed view (lossy-tolerant: missing
    /// fields fall back to defaults, matching the previous `.get(...)`
    /// behavior).
    pub fn from_value(v: &serde_json::Value) -> Self {
        serde_json::from_value(v.clone()).unwrap_or_default()
    }

    /// Lower into the canonical [`ChatResponse`]. The raw `chatcmpl-...`
    /// id is preserved; surface-specific id reformatting happens in the
    /// per-surface serializer.
    pub fn into_ir(self, model: String) -> ChatResponse {
        let choice = self.choices.into_iter().next().unwrap_or_default();
        let msg = choice.message;
        let reasoning = msg
            .reasoning_content
            .or(msg.reasoning)
            .filter(|s| !s.is_empty());
        let tool_calls = msg
            .tool_calls
            .into_iter()
            .map(|tc| ToolCall {
                id: tc.id,
                name: tc.function.name,
                arguments: serde_json::from_str(&tc.function.arguments)
                    .unwrap_or_else(|_| serde_json::Value::Object(Default::default())),
            })
            .collect();
        ChatResponse {
            id: self.id,
            model,
            content: msg.content.unwrap_or_default(),
            reasoning,
            tool_calls,
            finish_reason: FinishReason::from_openai(
                choice.finish_reason.as_deref().unwrap_or("stop"),
            ),
            usage: Usage {
                prompt_tokens: self.usage.prompt_tokens,
                completion_tokens: self.usage.completion_tokens,
            },
        }
    }
}
