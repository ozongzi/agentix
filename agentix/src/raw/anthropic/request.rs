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
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

/// Content payload for a `tool_result` block.
/// Anthropic accepts either a plain string or an array of text/image parts.
#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Parts(Vec<ToolResultPart>),
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultPart {
    Text { text: String },
    Image { source: ImageSource },
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

/// Merge consecutive `Message::User` entries into one by concatenating their
/// content parts. Anthropic requires strict user/assistant alternation.
fn merge_consecutive_user(messages: &[Message]) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::with_capacity(messages.len());
    for msg in messages {
        if let Message::User(parts) = msg
            && let Some(Message::User(prev)) = out.last_mut()
        {
            prev.extend(parts.iter().cloned());
            continue;
        }
        out.push(msg.clone());
    }
    out
}

pub(crate) fn build_anthropic_request(
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
    stream: bool,
) -> Request {
    let messages = merge_consecutive_user(messages);
    let mut out_messages: Vec<RequestMessage> = Vec::new();
    let mut pending_tool_results: Vec<ContentBlock> = Vec::new();

    for msg in &messages {
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
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
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
                    if let Some(t) = content
                        && !t.is_empty()
                    {
                        blocks.push(ContentBlock::Text {
                            text: t.clone(),
                            cache_control: None,
                        });
                    }
                    for tc in tool_calls {
                        let input = serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);
                        blocks.push(ContentBlock::ToolUse {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            input,
                            cache_control: None,
                        });
                    }
                    out_messages.push(RequestMessage {
                        role: "assistant",
                        content: MessageContent::Blocks(blocks),
                    });
                }
            }
            Message::ToolResult { call_id, content } => {
                use crate::request::Content;
                let wire_content = if let [Content::Text { text }] = content.as_slice() {
                    ToolResultContent::Text(text.clone())
                } else {
                    let parts = content
                        .iter()
                        .map(|p| match p {
                            Content::Text { text } => ToolResultPart::Text { text: text.clone() },
                            Content::Image(img) => {
                                let source = match &img.data {
                                    ImageData::Base64(data) => ImageSource::Base64 {
                                        media_type: img.mime_type.clone(),
                                        data: data.clone(),
                                    },
                                    ImageData::Url(url) => ImageSource::Url { url: url.clone() },
                                };
                                ToolResultPart::Image { source }
                            }
                        })
                        .collect();
                    ToolResultContent::Parts(parts)
                };
                pending_tool_results.push(ContentBlock::ToolResult {
                    tool_use_id: call_id.clone(),
                    content: wire_content,
                    cache_control: None,
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
    append_reminder(&mut out_messages, config.reminder.as_deref());

    let anthropic_tools: Option<Vec<Tool>> = if tools.is_empty() {
        None
    } else {
        Some(
            tools
                .iter()
                .map(|t| Tool {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    input_schema: t.function.parameters.clone(),
                })
                .collect(),
        )
    };

    let tool_choice = if tools.is_empty() {
        None
    } else {
        Some(ToolChoice::Auto)
    };

    // System prompt as blocks so we can attach cache_control.
    let system = config
        .system_prompt
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|s| {
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
    let mut blocks: Vec<ContentBlock> = parts
        .into_iter()
        .map(|p| match p {
            UserContent::Text { text: t } => ContentBlock::Text {
                text: t,
                cache_control: None,
            },
            UserContent::Image(img) => ContentBlock::Image {
                source: match img.data {
                    ImageData::Base64(data) => ImageSource::Base64 {
                        media_type: img.mime_type,
                        data,
                    },
                    ImageData::Url(url) => ImageSource::Url { url },
                },
                cache_control: None,
            },
        })
        .collect();
    if has_image && !has_text {
        blocks.push(ContentBlock::Text {
            text: " ".to_string(),
            cache_control: None,
        });
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

fn stamp_cache_breakpoints(messages: &mut [RequestMessage]) {
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

fn append_reminder(messages: &mut Vec<RequestMessage>, reminder: Option<&str>) {
    let Some(reminder) = reminder.filter(|s| !s.is_empty()) else {
        return;
    };
    let block = ContentBlock::Text {
        text: reminder.to_string(),
        cache_control: None,
    };
    if let Some(msg) = messages.last_mut().filter(|msg| msg.role == "user") {
        match &mut msg.content {
            MessageContent::Text(text) => {
                msg.content = MessageContent::Blocks(vec![
                    ContentBlock::Text {
                        text: text.clone(),
                        cache_control: None,
                    },
                    block,
                ]);
            }
            MessageContent::Blocks(blocks) => blocks.push(block),
        }
    } else {
        messages.push(RequestMessage {
            role: "user",
            content: MessageContent::Blocks(vec![block]),
        });
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
            if let Some(block) = blocks.last_mut() {
                set_cache_control(block);
            }
        }
    }
}

fn set_cache_control(block: &mut ContentBlock) {
    match block {
        ContentBlock::Text { cache_control, .. }
        | ContentBlock::Image { cache_control, .. }
        | ContentBlock::ToolUse { cache_control, .. }
        | ContentBlock::ToolResult { cache_control, .. } => {
            *cache_control = Some(CacheControl::ephemeral());
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
        let user = json["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "user")
            .unwrap()
            .clone();
        // After stamping, content becomes an array.
        let blocks = user["content"]
            .as_array()
            .expect("must be blocks after stamp");
        let text_block = blocks.iter().find(|b| b["type"] == "text").unwrap();
        assert_eq!(text_block["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn multi_turn_first_and_last_stamped() {
        let msgs = vec![
            Message::User(vec![Content::text("First")]),
            Message::Assistant {
                content: Some("A".into()),
                reasoning: None,
                tool_calls: vec![],
            },
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
            assert_eq!(
                text["cache_control"]["type"], "ephemeral",
                "both user messages must be stamped"
            );
        }
    }

    #[test]
    fn no_system_prompt_no_system_field() {
        let config = AgentConfig {
            model: "m".into(),
            ..Default::default()
        };
        let req = build_anthropic_request(&config, &[], &[], false);
        let json = serde_json::to_value(&req).unwrap();
        assert!(
            json["system"].is_null(),
            "absent system prompt must not serialize"
        );
    }

    #[test]
    fn reminder_is_after_last_user_breakpoint() {
        let mut config = cfg("S");
        config.reminder = Some("<runtime_context>plan</runtime_context>".into());
        let msgs = vec![Message::User(vec![Content::text("Actual user")])];
        let req = build_anthropic_request(&config, &msgs, &[], false);
        let json = serde_json::to_value(&req).unwrap();
        let messages = json["messages"].as_array().unwrap();
        let blocks = messages[0]["content"].as_array().unwrap();

        assert_eq!(blocks[0]["text"], "Actual user");
        assert_eq!(blocks[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(blocks[1]["text"], "<runtime_context>plan</runtime_context>");
        assert!(blocks[1]["cache_control"].is_null());
    }

    #[test]
    fn reminder_after_tool_result_keeps_tool_result_breakpoint() {
        let mut config = cfg("S");
        config.reminder = Some("<runtime_context>plan</runtime_context>".into());
        let msgs = vec![Message::ToolResult {
            call_id: "toolu_1".into(),
            content: vec![Content::text("579")],
        }];
        let req = build_anthropic_request(&config, &msgs, &[], false);
        let json = serde_json::to_value(&req).unwrap();
        let messages = json["messages"].as_array().unwrap();
        let blocks = messages[0]["content"].as_array().unwrap();

        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[1]["text"], "<runtime_context>plan</runtime_context>");
        assert!(blocks[1]["cache_control"].is_null());
    }
}
