use serde::Serialize;

use crate::config::AgentConfig;
use crate::raw::shared::ToolDefinition;
use crate::request::{ImageData, Message, ToolChoice, UserContent};

#[derive(Debug, Serialize)]
pub struct Request {
    pub model: String,
    pub messages: Vec<OaiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<crate::raw::shared::ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<crate::request::ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<crate::raw::shared::ResponseFormat>,
    #[serde(flatten)]
    pub extra_body: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum OaiMessage {
    System {
        content: MessageContent,
    },
    User {
        content: MessageContent,
    },
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ResponseToolCall>,
    },
    Tool {
        tool_call_id: String,
        content: ToolMessageContent,
    },
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ToolMessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ImageUrl {
        image_url: ImageUrl,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

#[derive(Debug, Serialize, Clone)]
pub struct CacheControl {
    pub r#type: String,
}

impl CacheControl {
    fn ephemeral() -> Self {
        CacheControl { r#type: "ephemeral".to_string() }
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ResponseToolCall {
    pub id: String,
    pub r#type: String,
    pub function: ResponseFunctionCall,
}

#[derive(Debug, Serialize, Clone)]
pub struct ResponseFunctionCall {
    pub name: String,
    pub arguments: String,
}

pub(crate) fn build_openrouter_request(
    config: &AgentConfig,
    history: Vec<Message>,
    tools: &[ToolDefinition],
    tool_choice: Option<ToolChoice>,
    stream: bool,
) -> Request {
    let mut messages = Vec::new();
    if let Some(s) = &config.system_prompt
        && !s.is_empty()
    {
        messages.push(OaiMessage::System {
            content: MessageContent::Text(s.clone()),
        });
    }
    for m in history {
        match m {
            Message::User(parts) => messages.push(OaiMessage::User {
                content: user_content_from_parts(parts),
            }),
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                messages.push(OaiMessage::Assistant {
                    content,
                    tool_calls: tool_calls
                        .into_iter()
                        .map(|tc| ResponseToolCall {
                            id: tc.id,
                            r#type: "function".to_string(),
                            function: ResponseFunctionCall {
                                name: tc.name,
                                arguments: tc.arguments,
                            },
                        })
                        .collect(),
                });
            }
            Message::ToolResult { call_id, content } => {
                use crate::request::Content;
                let tool_content = if let [Content::Text { text }] = content.as_slice() {
                    ToolMessageContent::Text(text.clone())
                } else {
                    let parts = content.iter().filter_map(|p| match p {
                        Content::Text { text } => Some(ContentPart::Text { text: text.clone(), cache_control: None }),
                        Content::Image(img) => {
                            let url = match &img.data {
                                ImageData::Base64(b) => format!("data:{};base64,{}", img.mime_type, b),
                                ImageData::Url(u) => u.clone(),
                            };
                            Some(ContentPart::ImageUrl { image_url: ImageUrl { url, detail: None }, cache_control: None })
                        }
                    }).collect();
                    ToolMessageContent::Parts(parts)
                };
                messages.push(OaiMessage::Tool {
                    tool_call_id: call_id.clone(),
                    content: tool_content,
                });
            }
        }
    }

    stamp_cache_breakpoints(&mut messages);

    let tools_opt = if tools.is_empty() {
        None
    } else {
        Some(tools.to_vec())
    };
    let extra = if config.extra_body.is_empty() {
        None
    } else {
        Some(config.extra_body.clone())
    };
    Request {
        model: config.model.clone(),
        messages,
        tools: tools_opt,
        tool_choice,
        stream: Some(stream),
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        response_format: config
            .response_format
            .clone()
            .map(crate::raw::shared::ResponseFormat::from),
        extra_body: extra,
    }
}

fn user_content_from_parts(parts: Vec<UserContent>) -> MessageContent {
    let blocks: Vec<ContentPart> = parts
        .into_iter()
        .map(|p| match p {
            UserContent::Text { text: t } => ContentPart::Text {
                text: t,
                cache_control: None,
            },
            UserContent::Image(img) => {
                let url = match img.data {
                    ImageData::Url(u) => u,
                    ImageData::Base64(b) => format!("data:{};base64,{}", img.mime_type, b),
                };
                ContentPart::ImageUrl {
                    image_url: ImageUrl { url, detail: None },
                    cache_control: None,
                }
            }
        })
        .collect();

    if let [ContentPart::Text { text, .. }] = blocks.as_slice() {
        return MessageContent::Text(text.clone());
    }
    MessageContent::Parts(blocks)
}

// Stamp cache_control: ephemeral on system prompt, first user message (summary), and last user message (latest turn).
fn stamp_cache_breakpoints(messages: &mut Vec<OaiMessage>) {
    let mut first_user: Option<usize> = None;
    let mut last_user: Option<usize> = None;

    for (i, msg) in messages.iter_mut().enumerate() {
        match msg {
            OaiMessage::System { content } => stamp_cache(content),
            OaiMessage::User { .. } => {
                first_user.get_or_insert(i);
                last_user = Some(i);
            }
            _ => {}
        }
    }

    if let Some(f) = first_user {
        if let OaiMessage::User { content } = &mut messages[f] {
            stamp_cache(content);
        }
        if let Some(l) = last_user.filter(|&l| l != f) {
            if let OaiMessage::User { content } = &mut messages[l] {
                stamp_cache(content);
            }
        }
    }
}

fn stamp_cache(content: &mut MessageContent) {
    match content {
        MessageContent::Text(text) => {
            *content = MessageContent::Parts(vec![ContentPart::Text {
                text: text.clone(),
                cache_control: Some(CacheControl::ephemeral()),
            }]);
        }
        MessageContent::Parts(parts) => {
            if let Some(last) = parts.last_mut() {
                set_cache_control(last);
            }
        }
    }
}

fn set_cache_control(part: &mut ContentPart) {
    match part {
        ContentPart::Text { cache_control, .. }
        | ContentPart::ImageUrl { cache_control, .. } => {
            *cache_control = Some(CacheControl::ephemeral());
        }
    }
}
