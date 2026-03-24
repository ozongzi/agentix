use serde::Serialize;

use crate::request::{ImageData, Message, UserContent};

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
    pub response_format: Option<crate::request::ResponseFormat>,
    #[serde(flatten)]
    pub extra_body: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum OaiMessage {
    System { content: String },
    User   { content: UserMessageContent },
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ResponseToolCall>,
    },
    Tool { tool_call_id: String, content: String },
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum UserMessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
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

impl From<crate::request::Request> for Request {
    fn from(req: crate::request::Request) -> Self {
        let mut messages = Vec::new();
        if let Some(s) = req.system_message {
            messages.push(OaiMessage::System { content: s });
        }
        for m in req.messages {
            match m {
                Message::User(parts) => messages.push(OaiMessage::User {
                    content: user_content_from_parts(parts),
                }),
                Message::Assistant { content, reasoning, tool_calls } => {
                    messages.push(OaiMessage::Assistant {
                        content,
                        reasoning_content: reasoning,
                        tool_calls: tool_calls.into_iter().map(|tc| ResponseToolCall {
                            id: tc.id,
                            r#type: "function".to_string(),
                            function: ResponseFunctionCall { name: tc.name, arguments: tc.arguments },
                        }).collect(),
                    });
                }
                Message::ToolResult { call_id, content } => messages.push(OaiMessage::Tool {
                    tool_call_id: call_id,
                    content,
                }),
            }
        }
        Request {
            model: req.model,
            messages,
            tools: req.tools,
            tool_choice: req.tool_choice,
            stream: Some(req.stream),
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            response_format: req.response_format,
            extra_body: req.extra_body,
        }
    }
}

fn user_content_from_parts(parts: Vec<UserContent>) -> UserMessageContent {
    let mut blocks: Vec<ContentPart> = parts.into_iter().map(|p| match p {
        UserContent::Text(t) => ContentPart::Text { text: t },
        UserContent::Image(img) => {
            let url = match img.data {
                ImageData::Url(u)    => u,
                ImageData::Base64(b) => format!("data:{};base64,{}", img.mime_type, b),
            };
            ContentPart::ImageUrl { image_url: ImageUrl { url, detail: None } }
        }
    }).collect();

    if blocks.len() == 1
        && let ContentPart::Text { text } = blocks.remove(0) {
            return UserMessageContent::Text(text);
        }
    UserMessageContent::Parts(blocks)
}
