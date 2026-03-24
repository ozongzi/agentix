use serde::Serialize;
use serde_json::Value;

use crate::raw::shared::ToolDefinition;
use crate::request::{ImageData, Message, UserContent};

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

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },
    Image { source: ImageSource },
    ToolUse { id: String, name: String, input: Value },
    ToolResult { tool_use_id: String, content: String },
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

impl From<crate::request::Request> for Request {
    fn from(req: crate::request::Request) -> Self {
        let mut messages: Vec<RequestMessage> = Vec::new();
        let mut pending_tool_results: Vec<ContentBlock> = Vec::new();

        for msg in req.messages {
            match msg {
                Message::User(parts) => {
                    if !pending_tool_results.is_empty() {
                        messages.push(RequestMessage {
                            role: "user",
                            content: MessageContent::Blocks(std::mem::take(&mut pending_tool_results)),
                        });
                    }
                    messages.push(RequestMessage {
                        role: "user",
                        content: user_content_from_parts(parts),
                    });
                }
                Message::Assistant { content, tool_calls, .. } => {
                    if tool_calls.is_empty() {
                        messages.push(RequestMessage {
                            role: "assistant",
                            content: MessageContent::Text(content.unwrap_or_default()),
                        });
                    } else {
                        let mut blocks: Vec<ContentBlock> = Vec::new();
                        if let Some(t) = content && !t.is_empty() {
                            blocks.push(ContentBlock::Text { text: t });
                        }
                        for tc in tool_calls {
                            let input = serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);
                            blocks.push(ContentBlock::ToolUse { id: tc.id, name: tc.name, input });
                        }
                        messages.push(RequestMessage {
                            role: "assistant",
                            content: MessageContent::Blocks(blocks),
                        });
                    }
                }
                Message::ToolResult { call_id, content } => {
                    pending_tool_results.push(ContentBlock::ToolResult {
                        tool_use_id: call_id,
                        content,
                    });
                }
            }
        }
        if !pending_tool_results.is_empty() {
            messages.push(RequestMessage {
                role: "user",
                content: MessageContent::Blocks(pending_tool_results),
            });
        }

        let tools = req.tools.map(|ts| {
            ts.into_iter().map(|t: ToolDefinition| Tool {
                name: t.function.name,
                description: t.function.description,
                input_schema: t.function.parameters,
            }).collect()
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
    let blocks = parts.into_iter().map(|p| match p {
        UserContent::Text(t) => ContentBlock::Text { text: t },
        UserContent::Image(img) => ContentBlock::Image {
            source: match img.data {
                ImageData::Base64(data) => ImageSource::Base64 { media_type: img.mime_type, data },
                ImageData::Url(url)     => ImageSource::Url { url },
            },
        },
    }).collect();
    MessageContent::Blocks(blocks)
}
