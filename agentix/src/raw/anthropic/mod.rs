//! Anthropic Messages API wire format.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::raw::shared::ToolDefinition;
use crate::request::{ImageData, Message, UserContent};

// ── Request ──────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct Request {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<RequestMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
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

/// Anthropic content — either a plain string or an array of blocks.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
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

#[derive(Debug, Serialize)]
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

// ── From<AgentRequest> ───────────────────────────────────────────────────────

impl From<crate::request::Request> for Request {
    fn from(req: crate::request::Request) -> Self {
        let mut messages: Vec<RequestMessage> = Vec::new();
        let mut pending_tool_results: Vec<ContentBlock> = Vec::new();

        for msg in req.messages {
            match msg {
                Message::User(parts) => {
                    // Flush any pending tool results first (same user turn)
                    if !pending_tool_results.is_empty() {
                        messages.push(RequestMessage {
                            role: "user",
                            content: MessageContent::Blocks(std::mem::take(
                                &mut pending_tool_results,
                            )),
                        });
                    }
                    let content = user_content_from_parts(parts);
                    messages.push(RequestMessage {
                        role: "user",
                        content,
                    });
                }
                Message::Assistant {
                    content,
                    tool_calls,
                    ..
                } => {
                    if tool_calls.is_empty() {
                        messages.push(RequestMessage {
                            role: "assistant",
                            content: MessageContent::Text(content.unwrap_or_default()),
                        });
                    } else {
                        let mut blocks: Vec<ContentBlock> = Vec::new();
                        if let Some(t) = content
                            && !t.is_empty()
                        {
                            blocks.push(ContentBlock::Text { text: t });
                        }
                        for tc in tool_calls {
                            let input = serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);
                            blocks.push(ContentBlock::ToolUse {
                                id: tc.id,
                                name: tc.name,
                                input,
                            });
                        }
                        messages.push(RequestMessage {
                            role: "assistant",
                            content: MessageContent::Blocks(blocks),
                        });
                    }
                }
                Message::ToolResult { call_id, content } => {
                    // Accumulate tool results — they'll be batched into one user message
                    pending_tool_results.push(ContentBlock::ToolResult {
                        tool_use_id: call_id,
                        content,
                    });
                }
            }
        }

        // Flush any remaining tool results
        if !pending_tool_results.is_empty() {
            messages.push(RequestMessage {
                role: "user",
                content: MessageContent::Blocks(pending_tool_results),
            });
        }

        let tools: Option<Vec<Tool>> = req.tools.map(|ts| {
            ts.into_iter()
                .map(|t: ToolDefinition| Tool {
                    name: t.function.name,
                    description: t.function.description,
                    input_schema: t.function.parameters,
                })
                .collect()
        });

        let tool_choice = req.tool_choice.map(|tc| match tc {
            crate::request::ToolChoice::Auto | crate::request::ToolChoice::None => ToolChoice::Auto,
            crate::request::ToolChoice::Required => ToolChoice::Any,
            crate::request::ToolChoice::Tool(name) => ToolChoice::Tool { name },
        });

        Request {
            model: req.model,
            max_tokens: req.max_tokens.unwrap_or(8192),
            messages,
            system: req.system_message.filter(|s| !s.is_empty()),
            tools,
            tool_choice,
            stream: Some(req.stream),
            temperature: req.temperature,
        }
    }
}

fn user_content_from_parts(parts: Vec<UserContent>) -> MessageContent {
    if parts.len() == 1 && matches!(&parts[0], UserContent::Text(_)) {
        if let UserContent::Text(t) = parts.into_iter().next().unwrap() {
            return MessageContent::Text(t);
        }
        unreachable!()
    }
    let blocks = parts
        .into_iter()
        .map(|p| match p {
            UserContent::Text(t) => ContentBlock::Text { text: t },
            UserContent::Image(img) => ContentBlock::Image {
                source: match img.data {
                    ImageData::Base64(data) => ImageSource::Base64 {
                        media_type: img.mime_type,
                        data,
                    },
                    ImageData::Url(url) => ImageSource::Url { url },
                },
            },
        })
        .collect();
    MessageContent::Blocks(blocks)
}

// ── Response ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct Response {
    pub id: String,
    pub content: Vec<ResponseBlock>,
    pub stop_reason: Option<String>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    Thinking {
        thinking: String,
    },
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl From<Usage> for crate::types::UsageStats {
    fn from(u: Usage) -> Self {
        Self {
            prompt_tokens: u.input_tokens as usize,
            completion_tokens: u.output_tokens as usize,
            total_tokens: (u.input_tokens + u.output_tokens) as usize,
        }
    }
}

// ── Streaming events ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart {
        message: MessageStart,
    },
    ContentBlockStart {
        index: u32,
        content_block: ContentBlockStart,
    },
    ContentBlockDelta {
        index: u32,
        delta: ContentBlockDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: MessageDelta,
        usage: Option<Usage>,
    },
    MessageStop,
    Error {
        error: StreamError,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
pub struct MessageStart {
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockStart {
    Text { text: String },
    ToolUse { id: String, name: String },
    Thinking { thinking: String },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
}

#[derive(Debug, Deserialize)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct StreamError {
    pub r#type: String,
    pub message: String,
}

// ── ProviderProtocol ─────────────────────────────────────────────────────────

use crate::markers::Anthropic;
use crate::request::ToolCall as AgentToolCall;
use crate::types::{ProtocolEvent, PartialToolCall, ProviderProtocol, StreamBufs, ToolCallChunk};

impl ProviderProtocol for Anthropic {
    type RawRequest = Request;
    type RawResponse = Response;
    type RawChunk = StreamEvent;

    fn build_raw(req: crate::request::Request) -> Request {
        Request::from(req)
    }
    fn default_base_url() -> &'static str {
        "https://api.anthropic.com"
    }

    fn parse_response(raw: Response) -> (Vec<ProtocolEvent>, Vec<AgentToolCall>) {
        let mut events = Vec::new();
        if let Some(u) = raw.usage {
            events.push(ProtocolEvent::Usage(u.into()));
        }
        let mut tool_calls = Vec::new();
        for block in raw.content {
            match block {
                ResponseBlock::Thinking { thinking } if !thinking.is_empty() => {
                    events.push(ProtocolEvent::Reasoning(thinking));
                }
                ResponseBlock::Text { text } if !text.is_empty() => {
                    events.push(ProtocolEvent::Token(text));
                }
                ResponseBlock::ToolUse { id, name, input } => {
                    tool_calls.push(AgentToolCall {
                        id,
                        name,
                        arguments: serde_json::to_string(&input).unwrap_or_default(),
                    });
                }
                _ => {}
            }
        }
        (events, tool_calls)
    }

    fn parse_chunk(chunk: StreamEvent, bufs: &mut StreamBufs) -> Vec<ProtocolEvent> {
        match chunk {
            StreamEvent::MessageStart { message } => {
                if let Some(u) = message.usage {
                    return vec![ProtocolEvent::Usage(u.into())];
                }
                vec![]
            }
            StreamEvent::MessageDelta { usage, .. } => {
                if let Some(u) = usage {
                    return vec![ProtocolEvent::Usage(u.into())];
                }
                vec![]
            }
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                let idx = index as usize;
                if bufs.tool_call_bufs.len() <= idx {
                    bufs.tool_call_bufs.resize_with(idx + 1, || None);
                }
                match content_block {
                    ContentBlockStart::ToolUse { id, name } => {
                        bufs.tool_call_bufs[idx] = Some(PartialToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            arguments: String::new(),
                        });
                        vec![ProtocolEvent::ToolCallChunk(ToolCallChunk {
                            id,
                            name,
                            delta: String::new(),
                            index,
                        })]
                    }
                    ContentBlockStart::Text { text } if !text.is_empty() => {
                        bufs.content_buf.push_str(&text);
                        vec![ProtocolEvent::Token(text)]
                    }
                    ContentBlockStart::Thinking { thinking } if !thinking.is_empty() => {
                        bufs.reasoning_buf.push_str(&thinking);
                        vec![ProtocolEvent::Reasoning(thinking)]
                    }
                    _ => vec![],
                }
            }
            StreamEvent::ContentBlockDelta { index, delta } => match delta {
                ContentBlockDelta::TextDelta { text } if !text.is_empty() => {
                    bufs.content_buf.push_str(&text);
                    vec![ProtocolEvent::Token(text)]
                }
                ContentBlockDelta::ThinkingDelta { thinking } if !thinking.is_empty() => {
                    bufs.reasoning_buf.push_str(&thinking);
                    vec![ProtocolEvent::Reasoning(thinking)]
                }
                ContentBlockDelta::InputJsonDelta { partial_json } if !partial_json.is_empty() => {
                    let idx = index as usize;
                    if let Some(Some(partial)) = bufs.tool_call_bufs.get_mut(idx) {
                        partial.arguments.push_str(&partial_json);
                        vec![ProtocolEvent::ToolCallChunk(ToolCallChunk {
                            id: partial.id.clone(),
                            name: partial.name.clone(),
                            delta: partial_json,
                            index,
                        })]
                    } else {
                        vec![]
                    }
                }
                _ => vec![],
            },
            _ => vec![],
        }
    }

    fn finalize_stream(bufs: &mut StreamBufs) -> Vec<AgentToolCall> {
        bufs.tool_call_bufs
            .drain(..)
            .flatten()
            .map(|p| AgentToolCall {
                id: p.id,
                name: p.name,
                arguments: p.arguments,
            })
            .collect()
    }

    fn url_suffix(_model: &str, _streaming: bool) -> String {
        "/v1/messages".to_string()
    }

    fn extra_headers() -> &'static [(&'static str, &'static str)] {
        &[("anthropic-version", "2023-06-01")]
    }

    fn auth_header_name() -> Option<&'static str> {
        Some("x-api-key")
    }
}
