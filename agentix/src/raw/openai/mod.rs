//! OpenAI-compatible chat completions wire format.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::raw::shared::{ResponseFormat, ToolChoice, ToolDefinition};
use crate::request::{ImageData, Message, UserContent};

// ── Request ──────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Default)]
pub struct Request {
    pub model: String,
    pub messages: Vec<RequestMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<Map<String, Value>>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum RequestMessage {
    System {
        content: String,
    },
    User {
        content: UserMessageContent,
    },
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
    },
    Tool {
        content: String,
        tool_call_id: String,
    },
}

/// User message content — either a plain string or a multimodal array.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum UserMessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Serialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: &'static str,
    pub function: FunctionCall,
}

#[derive(Debug, Serialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

// ResponseFormat is imported from crate::raw::shared

// ── From<AgentRequest> ───────────────────────────────────────────────────────

impl From<crate::request::Request> for Request {
    fn from(req: crate::request::Request) -> Self {
        let mut messages: Vec<RequestMessage> = Vec::new();

        if let Some(sys) = req.system_message
            && !sys.is_empty()
        {
            messages.push(RequestMessage::System { content: sys });
        }

        for msg in req.messages {
            match msg {
                Message::User(parts) => {
                    let content = user_content_from_parts(parts);
                    messages.push(RequestMessage::User { content });
                }
                Message::Assistant {
                    content,
                    reasoning,
                    tool_calls,
                } => {
                    messages.push(RequestMessage::Assistant {
                        content,
                        reasoning_content: reasoning,
                        tool_calls: if tool_calls.is_empty() {
                            None
                        } else {
                            Some(
                                tool_calls
                                    .into_iter()
                                    .map(|tc| ToolCall {
                                        id: tc.id,
                                        r#type: "function",
                                        function: FunctionCall {
                                            name: tc.name,
                                            arguments: tc.arguments,
                                        },
                                    })
                                    .collect(),
                            )
                        },
                    });
                }
                Message::ToolResult { call_id, content } => {
                    messages.push(RequestMessage::Tool {
                        content,
                        tool_call_id: call_id,
                    });
                }
            }
        }

        let tool_choice = req.tool_choice.map(Into::into);
        let response_format = req.response_format.map(Into::into);

        Request {
            model: req.model,
            messages,
            tools: req.tools,
            tool_choice,
            stream: Some(req.stream),
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            response_format,
            extra_body: req.extra_body,
        }
    }
}

fn user_content_from_parts(parts: Vec<UserContent>) -> UserMessageContent {
    if parts.len() == 1
        && matches!(&parts[0], UserContent::Text(_))
        && !matches!(&parts[0], UserContent::Image(_))
    {
        if let UserContent::Text(t) = parts.into_iter().next().unwrap() {
            return UserMessageContent::Text(t);
        }
        unreachable!()
    }
    let blocks = parts
        .into_iter()
        .map(|p| match p {
            UserContent::Text(t) => ContentPart::Text { text: t },
            UserContent::Image(img) => {
                let url = match img.data {
                    ImageData::Url(u) => u,
                    ImageData::Base64(b) => format!("data:{};base64,{}", img.mime_type, b),
                };
                ContentPart::ImageUrl {
                    image_url: ImageUrl { url, detail: None },
                }
            }
        })
        .collect();
    UserMessageContent::Parts(blocks)
}

// ── Response ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct Response {
    pub id: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ResponseMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ResponseToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct ResponseToolCall {
    pub id: String,
    pub function: ResponseFunctionCall,
}

#[derive(Debug, Deserialize)]
pub struct ResponseFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Streaming chunk ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct StreamChunk {
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkChoice {
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Delta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct DeltaToolCall {
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<DeltaFunctionCall>,
}

#[derive(Debug, Deserialize)]
pub struct DeltaFunctionCall {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

// ── ProviderProtocol ─────────────────────────────────────────────────────────

use crate::markers::{DeepSeek, OpenAI};
use crate::request::ToolCall as AgentToolCall;
use crate::types::{AgentEvent, PartialToolCall, ProviderProtocol, StreamBufs, ToolCallChunk};

fn parse_openai_response(raw: Response) -> (Vec<AgentEvent>, Vec<AgentToolCall>) {
    let choice = match raw.choices.into_iter().next() {
        Some(c) => c,
        None => return (vec![], vec![]),
    };
    let msg = choice.message;
    let mut events = Vec::new();
    if let Some(r) = msg.reasoning_content.filter(|s| !s.is_empty()) {
        events.push(AgentEvent::ReasoningToken(r));
    }
    if let Some(t) = msg.content.filter(|s| !s.is_empty()) {
        events.push(AgentEvent::Token(t));
    }
    let tool_calls = msg
        .tool_calls
        .unwrap_or_default()
        .into_iter()
        .map(|tc| AgentToolCall {
            id: tc.id,
            name: tc.function.name,
            arguments: tc.function.arguments,
        })
        .collect();
    (events, tool_calls)
}

fn parse_openai_chunk(chunk: StreamChunk, bufs: &mut StreamBufs) -> Vec<AgentEvent> {
    let choice = match chunk.choices.into_iter().next() {
        Some(c) => c,
        None => return vec![],
    };
    let delta = choice.delta;
    if let Some(dtcs) = delta.tool_calls {
        let mut events = Vec::new();
        for dtc in dtcs {
            let idx = dtc.index as usize;
            if bufs.tool_call_bufs.len() <= idx {
                bufs.tool_call_bufs.resize_with(idx + 1, || None);
            }
            let entry = &mut bufs.tool_call_bufs[idx];
            if entry.is_none() {
                let id = dtc.id.clone().unwrap_or_default();
                let name = dtc
                    .function
                    .as_ref()
                    .and_then(|f| f.name.clone())
                    .unwrap_or_default();
                events.push(AgentEvent::ToolCall(ToolCallChunk {
                    id: id.clone(),
                    name: name.clone(),
                    delta: String::new(),
                    index: idx as u32,
                }));
                *entry = Some(PartialToolCall {
                    id,
                    name,
                    arguments: String::new(),
                });
            }
            if let Some(partial) = entry.as_mut() {
                if let Some(id) = dtc.id.filter(|_| partial.id.is_empty()) {
                    partial.id = id;
                }
                if let Some(args) = dtc
                    .function
                    .and_then(|f| f.arguments)
                    .filter(|a| !a.is_empty())
                {
                    partial.arguments.push_str(&args);
                    events.push(AgentEvent::ToolCall(ToolCallChunk {
                        id: partial.id.clone(),
                        name: partial.name.clone(),
                        delta: args,
                        index: idx as u32,
                    }));
                }
            }
        }
        return events;
    }
    if let Some(r) = delta.reasoning_content.filter(|s| !s.is_empty()) {
        bufs.reasoning_buf.push_str(&r);
        return vec![AgentEvent::ReasoningToken(r)];
    }
    if let Some(t) = delta.content.filter(|s| !s.is_empty()) {
        bufs.content_buf.push_str(&t);
        return vec![AgentEvent::Token(t)];
    }
    vec![]
}

fn finalize_openai_stream(bufs: &mut StreamBufs) -> Vec<AgentToolCall> {
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

impl ProviderProtocol for OpenAI {
    type RawRequest = Request;
    type RawResponse = Response;
    type RawChunk = StreamChunk;
    fn build_raw(req: crate::request::Request) -> Request {
        Request::from(req)
    }
    fn parse_response(raw: Response) -> (Vec<AgentEvent>, Vec<AgentToolCall>) {
        parse_openai_response(raw)
    }
    fn parse_chunk(chunk: StreamChunk, bufs: &mut StreamBufs) -> Vec<AgentEvent> {
        parse_openai_chunk(chunk, bufs)
    }
    fn finalize_stream(bufs: &mut StreamBufs) -> Vec<AgentToolCall> {
        finalize_openai_stream(bufs)
    }
    fn default_base_url() -> &'static str {
        "https://api.openai.com/v1"
    }
}

impl ProviderProtocol for DeepSeek {
    type RawRequest = Request;
    type RawResponse = Response;
    type RawChunk = StreamChunk;
    fn build_raw(req: crate::request::Request) -> Request {
        Request::from(req)
    }
    fn parse_response(raw: Response) -> (Vec<AgentEvent>, Vec<AgentToolCall>) {
        parse_openai_response(raw)
    }
    fn parse_chunk(chunk: StreamChunk, bufs: &mut StreamBufs) -> Vec<AgentEvent> {
        parse_openai_chunk(chunk, bufs)
    }
    fn finalize_stream(bufs: &mut StreamBufs) -> Vec<AgentToolCall> {
        finalize_openai_stream(bufs)
    }
    fn default_base_url() -> &'static str {
        "https://api.deepseek.com"
    }

    fn prepare_history(messages: Vec<crate::request::Message>) -> Vec<crate::request::Message> {
        use crate::request::Message;
        // deepseek-reasoner rules applied to the outgoing copy (stored history is untouched):
        //   assistant WITH tool_calls  → reasoning must be present (fill "" if absent)
        //   assistant WITHOUT tool_calls → reasoning must be None
        messages
            .into_iter()
            .map(|m| match m {
                Message::Assistant {
                    content,
                    reasoning,
                    tool_calls,
                } => {
                    let has_tools = !tool_calls.is_empty();
                    Message::Assistant {
                        content,
                        reasoning: if has_tools {
                            Some(reasoning.unwrap_or_default())
                        } else {
                            None
                        },
                        tool_calls,
                    }
                }
                other => other,
            })
            .collect()
    }
}
