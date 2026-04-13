use serde::Serialize;
use serde_json::Value;

use crate::config::AgentConfig;
use crate::raw::shared::ToolDefinition;
use crate::request::{ImageData, Message, UserContent};

// ── Cache control ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Clone)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub kind: &'static str,
}

impl CacheControl {
    fn ephemeral() -> Self {
        CacheControl { kind: "ephemeral" }
    }
}

// ── System prompt (block format required for cache_control) ───────────────────

#[derive(Debug, Serialize)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

// ── Request ───────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct Request {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<RequestMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<SystemBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct RequestMessage {
    pub role: &'static str,
    pub content: MessageContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Image {
        source: ImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

#[derive(Debug, Serialize)]
pub struct Tool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

pub(crate) fn build_anthropic_request(
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
    stream: bool,
) -> Request {
    let mut out_messages: Vec<RequestMessage> = Vec::new();
    let mut pending_tool_results: Vec<ContentBlock> = Vec::new();

    for msg in messages {
        match msg {
            Message::User(parts) => {
                if !pending_tool_results.is_empty() {
                    out_messages.push(RequestMessage {
                        role: "user",
                        content: MessageContent::Blocks(std::mem::take(&mut pending_tool_results)),
                    });
                }
                out_messages.push(RequestMessage {
                    role: "user",
                    content: user_content_from_parts(parts.clone()),
                });
            }
            Message::Assistant { content, tool_calls, .. } => {
                if !pending_tool_results.is_empty() {
                    out_messages.push(RequestMessage {
                        role: "user",
                        content: MessageContent::Blocks(std::mem::take(&mut pending_tool_results)),
                    });
                }
                if tool_calls.is_empty() {
                    out_messages.push(RequestMessage {
                        role: "assistant",
                        content: MessageContent::Text(content.clone().unwrap_or_default()),
                    });
                } else {
                    let mut blocks: Vec<ContentBlock> = Vec::new();
                    if let Some(t) = content && !t.is_empty() {
                        blocks.push(ContentBlock::Text { text: t.clone(), cache_control: None });
                    }
                    for tc in tool_calls {
                        let input = serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);
                        blocks.push(ContentBlock::ToolUse { id: tc.id.clone(), name: tc.name.clone(), input });
                    }
                    out_messages.push(RequestMessage {
                        role: "assistant",
                        content: MessageContent::Blocks(blocks),
                    });
                }
            }
            Message::ToolResult { call_id, content } => {
                use crate::raw::shared::{content_to_wire, ContentWire};
                let wire_content = match content_to_wire(content) {
                    ContentWire::Text(t) => t.to_string(),
                    ContentWire::Parts(parts) => serde_json::to_string(parts)
                        .unwrap_or_default(),
                };
                pending_tool_results.push(ContentBlock::ToolResult {
                    tool_use_id: call_id.clone(),
                    content: wire_content,
                });
            }
        }
    }
    if !pending_tool_results.is_empty() {
        out_messages.push(RequestMessage {
            role: "user",
            content: MessageContent::Blocks(pending_tool_results),
        });
    }

    stamp_cache_breakpoints(&mut out_messages);

    let anthropic_tools: Option<Vec<Tool>> = if tools.is_empty() {
        None
    } else {
        Some(tools.iter().map(|t| Tool {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            input_schema: t.function.parameters.clone(),
        }).collect())
    };

    let tool_choice = if tools.is_empty() {
        None
    } else {
        Some(ToolChoice::Auto)
    };

    // System prompt as blocks so we can attach cache_control.
    let system = config.system_prompt.as_deref().filter(|s| !s.is_empty()).map(|s| {
        vec![SystemBlock {
            kind: "text",
            text: s.to_string(),
            cache_control: Some(CacheControl::ephemeral()),
        }]
    });

    Request {
        model: config.model.clone(),
        max_tokens: config.max_tokens.unwrap_or(32_768),
        messages: out_messages,
        system,
        tools: anthropic_tools,
        tool_choice,
        stream: Some(stream),
        temperature: config.temperature,
    }
}

fn user_content_from_parts(parts: Vec<UserContent>) -> MessageContent {
    if parts.len() == 1 && matches!(&parts[0], UserContent::Text { .. }) {
        if let UserContent::Text { text: t } = parts.into_iter().next().unwrap() {
            return MessageContent::Text(t);
        }
        unreachable!()
    }
    let has_text = parts.iter().any(|p| matches!(p, UserContent::Text { .. }));
    let has_image = parts.iter().any(|p| matches!(p, UserContent::Image(_)));
    let mut blocks: Vec<ContentBlock> = parts.into_iter().map(|p| match p {
        UserContent::Text { text: t } => ContentBlock::Text { text: t, cache_control: None },
        UserContent::Image(img) => ContentBlock::Image {
            source: match img.data {
                ImageData::Base64(data) => ImageSource::Base64 { media_type: img.mime_type, data },
                ImageData::Url(url)     => ImageSource::Url { url },
            },
        },
    }).collect();
    if has_image && !has_text {
        blocks.push(ContentBlock::Text { text: " ".to_string(), cache_control: None });
    }
    MessageContent::Blocks(blocks)
}

// ── Cache breakpoints ─────────────────────────────────────────────────────────
//
// Strategy (mirrors OpenRouter):
//   • first user message  → breakpoint (covers compact summary / stable history head)
//   • last  user message  → breakpoint (covers current turn; warms cache for next request)
//
// System prompt already gets cache_control in build_anthropic_request above.

fn stamp_cache_breakpoints(messages: &mut Vec<RequestMessage>) {
    let mut first_user: Option<usize> = None;
    let mut last_user: Option<usize> = None;

    for (i, msg) in messages.iter().enumerate() {
        if msg.role == "user" {
            first_user.get_or_insert(i);
            last_user = Some(i);
        }
    }

    if let Some(f) = first_user {
        stamp_cache(&mut messages[f].content);
        if let Some(l) = last_user.filter(|&l| l != f) {
            stamp_cache(&mut messages[l].content);
        }
    }
}

fn stamp_cache(content: &mut MessageContent) {
    match content {
        MessageContent::Text(text) => {
            *content = MessageContent::Blocks(vec![ContentBlock::Text {
                text: text.clone(),
                cache_control: Some(CacheControl::ephemeral()),
            }]);
        }
        MessageContent::Blocks(blocks) => {
            // Stamp on the last text block (skip ToolUse/ToolResult/Image).
            if let Some(block) = blocks.iter_mut().rev().find(|b| matches!(b, ContentBlock::Text { .. })) {
                if let ContentBlock::Text { cache_control, .. } = block {
                    *cache_control = Some(CacheControl::ephemeral());
                }
            }
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{Content, Message};

    fn cfg(system: &str) -> AgentConfig {
        AgentConfig {
            system_prompt: Some(system.into()),
            model: "claude-haiku-4-5-20251001".into(),
            ..Default::default()
        }
    }

    #[test]
    fn system_block_has_cache_control() {
        let req = build_anthropic_request(&cfg("Be helpful."), &[], &[], false);
        let json = serde_json::to_value(&req).unwrap();
        let blocks = json["system"].as_array().expect("system must be array");
        let last = blocks.last().unwrap();
        assert_eq!(last["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn single_user_message_gets_breakpoint() {
        let msgs = vec![Message::User(vec![Content::text("Hello")])];
        let req = build_anthropic_request(&cfg("S"), &msgs, &[], false);
        let json = serde_json::to_value(&req).unwrap();
        let user = json["messages"].as_array().unwrap()
            .iter().find(|m| m["role"] == "user").unwrap().clone();
        // After stamping, content becomes an array.
        let blocks = user["content"].as_array().expect("must be blocks after stamp");
        let text_block = blocks.iter().find(|b| b["type"] == "text").unwrap();
        assert_eq!(text_block["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn multi_turn_first_and_last_stamped() {
        let msgs = vec![
            Message::User(vec![Content::text("First")]),
            Message::Assistant { content: Some("A".into()), reasoning: None, tool_calls: vec![] },
            Message::User(vec![Content::text("Second")]),
        ];
        let req = build_anthropic_request(&cfg("S"), &msgs, &[], false);
        let json = serde_json::to_value(&req).unwrap();
        let messages = json["messages"].as_array().unwrap();

        let users: Vec<_> = messages.iter().filter(|m| m["role"] == "user").collect();
        assert_eq!(users.len(), 2);

        for u in &users {
            let blocks = u["content"].as_array().expect("must be blocks");
            let text = blocks.iter().find(|b| b["type"] == "text").unwrap();
            assert_eq!(text["cache_control"]["type"], "ephemeral",
                "both user messages must be stamped");
        }
    }

    #[test]
    fn no_system_prompt_no_system_field() {
        let config = AgentConfig { model: "m".into(), ..Default::default() };
        let req = build_anthropic_request(&config, &[], &[], false);
        let json = serde_json::to_value(&req).unwrap();
        assert!(json["system"].is_null(), "absent system prompt must not serialize");
    }
}
