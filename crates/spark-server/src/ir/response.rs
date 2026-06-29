// SPDX-License-Identifier: AGPL-3.0-only
//
// Canonical chat IR (response direction). The model's output is lowered
// into these model-agnostic types; each API surface serializes them back
// to its own wire format, replacing the untyped `serde_json::Value`
// reading that the Anthropic surface used to do.

use super::message::ToolCall;

/// A complete (non-streaming) assistant response.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub content: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

/// Prompt / completion token counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// Why generation stopped. `Other` preserves unknown wire reasons
/// losslessly (PCND: no silent default).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Other(String),
}

impl FinishReason {
    /// Map an OpenAI `finish_reason` string.
    pub fn from_openai(s: &str) -> Self {
        match s {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "tool_calls" => FinishReason::ToolCalls,
            "content_filter" => FinishReason::ContentFilter,
            other => FinishReason::Other(other.to_string()),
        }
    }

    /// The OpenAI `finish_reason` wire string.
    pub fn as_openai(&self) -> &str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::Other(s) => s,
        }
    }
}
