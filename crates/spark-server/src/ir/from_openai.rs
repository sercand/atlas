// SPDX-License-Identifier: AGPL-3.0-only
//
// Adapter: OpenAI Chat Completions wire message (`IncomingMessage`) →
// canonical `ir::Message`. This is the OpenAI request-direction edge of
// the narrow waist.

use super::message::{ContentPart, ImageData, ImageSource, Message, Reasoning, Role, ToolCall};
use crate::openai::IncomingMessage;

impl From<&IncomingMessage> for Message {
    fn from(m: &IncomingMessage) -> Self {
        // Images first (matching the template's historical `[image*N,
        // text]` content-array shape), then a single text part. The
        // original wire interleaving was already flattened away by
        // `ParsedContent`, and `build_msg_entries`/the template read
        // `text()` and `image_count()` independently, so part order here
        // does not affect rendering.
        let mut content: Vec<ContentPart> = Vec::new();
        for img in &m.content.images {
            content.push(ContentPart::Image(ImageSource {
                data: ImageData::Base64(img.clone()),
            }));
        }
        if !m.content.text.is_empty() {
            content.push(ContentPart::Text(m.content.text.clone()));
        }

        // Tool calls: parse the wire `arguments` string into structured
        // JSON exactly as the historical path did
        // (`from_str(..).unwrap_or(Object::default())`); a missing id
        // becomes the empty string, also matching today.
        let tool_calls: Vec<ToolCall> = m
            .tool_calls
            .as_ref()
            .map(|tcs| {
                tcs.iter()
                    .map(|tc| ToolCall {
                        id: tc.id.clone().unwrap_or_default(),
                        name: tc.function.name.clone(),
                        arguments: serde_json::from_str(&tc.function.arguments)
                            .unwrap_or_else(|_| serde_json::Value::Object(Default::default())),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Message {
            role: Role::from_wire_lossless(&m.role),
            content,
            tool_calls,
            tool_call_id: m.tool_call_id.clone(),
            name: m.name.clone(),
            reasoning: m.reasoning_content.clone().map(|text| Reasoning { text }),
            tool_error: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::message::{ContentPart, ImageData, ImageSource, Reasoning};
    use crate::openai::{IncomingMessage, ParsedContent};
    use crate::tool_parser::{IncomingFunction, IncomingToolCall};

    fn msg(role: &str) -> IncomingMessage {
        IncomingMessage {
            role: role.to_string(),
            content: ParsedContent::default(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn text_only_user_message() {
        let mut m = msg("user");
        m.content.text = "hi".into();
        let ir: Message = (&m).into();
        assert_eq!(ir.role, Role::User);
        assert_eq!(ir.content, vec![ContentPart::Text("hi".into())]);
        assert_eq!(ir.image_count(), 0);
        assert!(ir.tool_calls.is_empty());
        assert!(ir.reasoning.is_none());
        assert!(!ir.tool_error);
    }

    #[test]
    fn empty_content_yields_no_parts() {
        let m = msg("user");
        let ir: Message = (&m).into();
        assert!(ir.content.is_empty());
        assert_eq!(ir.text(), "");
    }

    #[test]
    fn images_precede_text_and_are_preserved_verbatim() {
        let mut m = msg("user");
        m.content.text = "see".into();
        m.content.images = vec!["data:image/png;base64,AAA".into()];
        let ir: Message = (&m).into();
        // Images first (matches the template's [image*N, text] order), then text.
        assert_eq!(
            ir.content,
            vec![
                ContentPart::Image(ImageSource {
                    data: ImageData::Base64("data:image/png;base64,AAA".into()),
                }),
                ContentPart::Text("see".into()),
            ]
        );
        assert_eq!(ir.image_count(), 1);
    }

    #[test]
    fn assistant_tool_calls_parse_arguments_to_json() {
        let mut m = msg("assistant");
        m.tool_calls = Some(vec![IncomingToolCall {
            id: Some("call_1".into()),
            function: IncomingFunction {
                name: "get_weather".into(),
                arguments: r#"{"city":"SF"}"#.into(),
            },
        }]);
        let ir: Message = (&m).into();
        assert_eq!(ir.role, Role::Assistant);
        assert_eq!(ir.tool_calls.len(), 1);
        assert_eq!(ir.tool_calls[0].id, "call_1");
        assert_eq!(ir.tool_calls[0].name, "get_weather");
        assert_eq!(
            ir.tool_calls[0].arguments,
            serde_json::json!({"city": "SF"})
        );
    }

    #[test]
    fn malformed_tool_args_default_to_empty_object() {
        // Mirrors the historical `from_str(...).unwrap_or(Object::default())`.
        let mut m = msg("assistant");
        m.tool_calls = Some(vec![IncomingToolCall {
            id: None,
            function: IncomingFunction {
                name: "f".into(),
                arguments: "not json".into(),
            },
        }]);
        let ir: Message = (&m).into();
        assert_eq!(ir.tool_calls[0].id, ""); // missing id → empty string (preserved)
        assert_eq!(ir.tool_calls[0].arguments, serde_json::json!({}));
    }

    #[test]
    fn reasoning_content_maps_to_first_class_reasoning() {
        let mut m = msg("assistant");
        m.reasoning_content = Some("ponder".into());
        let ir: Message = (&m).into();
        assert_eq!(
            ir.reasoning,
            Some(Reasoning {
                text: "ponder".into()
            })
        );

        let m2 = msg("assistant");
        let ir2: Message = (&m2).into();
        assert!(ir2.reasoning.is_none());
    }

    #[test]
    fn tool_message_preserves_call_id_and_name() {
        let mut m = msg("tool");
        m.content.text = "exit 0".into();
        m.tool_call_id = Some("call_1".into());
        m.name = Some("bash".into());
        let ir: Message = (&m).into();
        assert_eq!(ir.role, Role::Tool);
        assert_eq!(ir.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(ir.name.as_deref(), Some("bash"));
        assert_eq!(ir.text(), "exit 0");
    }

    #[test]
    fn unknown_role_is_preserved_losslessly() {
        let m = msg("developer");
        let ir: Message = (&m).into();
        assert_eq!(ir.role, Role::Other("developer".into()));
        assert_eq!(ir.role.as_wire(), "developer");
    }
}
