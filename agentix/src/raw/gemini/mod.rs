//! Google Gemini API wire format.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::raw::shared::ToolDefinition;
use crate::request::{ToolChoice, ImageData, Message, UserContent};

// ── Request ──────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Request {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTools>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
}

#[derive(Debug, Serialize)]
pub struct SystemInstruction {
    pub parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
pub struct Content {
    pub role: &'static str,
    pub parts: Vec<Part>,
}

pub enum Part {
    Text(String),
    InlineData(Blob),
    FunctionCall(FunctionCall),
    FunctionResponse(FunctionResponse),
}

impl std::fmt::Debug for Part {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Part::Text(t) => write!(f, "Text({t:?})"),
            Part::InlineData(_) => write!(f, "InlineData(...)"),
            Part::FunctionCall(fc) => write!(f, "FunctionCall({:?})", fc.name),
            Part::FunctionResponse(fr) => write!(f, "FunctionResponse({:?})", fr.name),
        }
    }
}

impl serde::Serialize for Part {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(None)?;
        match self {
            Part::Text(t) => {
                map.serialize_entry("text", t)?;
            }
            Part::InlineData(b) => {
                map.serialize_entry("inline_data", b)?;
            }
            Part::FunctionCall(fc) => {
                map.serialize_entry("function_call", fc)?;
            }
            Part::FunctionResponse(fr) => {
                map.serialize_entry("function_response", fr)?;
            }
        }
        map.end()
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Blob {
    pub mime_type: String,
    pub data: String, // base64
}

#[derive(Debug, Serialize)]
pub struct FunctionCall {
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Serialize)]
pub struct FunctionResponse {
    pub name: String,
    pub response: Value,
}

#[derive(Debug, Serialize)]
pub struct GeminiTools {
    pub function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Debug, Serialize)]
pub struct FunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    pub function_calling_config: FunctionCallingConfig,
}

#[derive(Debug, Serialize)]
pub struct FunctionCallingConfig {
    pub mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<&'static str>,
}

// ── From<AgentRequest> ───────────────────────────────────────────────────────

impl From<crate::request::Request> for Request {
    fn from(req: crate::request::Request) -> Self {
        let system_instruction =
            req.system_message
                .filter(|s| !s.is_empty())
                .map(|s| SystemInstruction {
                    parts: vec![Part::Text(s)],
                });

        let mut contents: Vec<Content> = Vec::new();
        let mut pending_fn_responses: Vec<Part> = Vec::new();

        for msg in req.messages {
            match msg {
                Message::User(parts) => {
                    if !pending_fn_responses.is_empty() {
                        contents.push(Content {
                            role: "user",
                            parts: std::mem::take(&mut pending_fn_responses),
                        });
                    }
                    let gparts: Vec<Part> = parts
                        .into_iter()
                        .map(|p| match p {
                            UserContent::Text(t) => Part::Text(t),
                            UserContent::Image(img) => Part::InlineData(Blob {
                                mime_type: img.mime_type,
                                data: match img.data {
                                    ImageData::Base64(b) => b,
                                    ImageData::Url(u) => u,
                                },
                            }),
                        })
                        .collect();
                    contents.push(Content {
                        role: "user",
                        parts: gparts,
                    });
                }
                Message::Assistant {
                    content,
                    tool_calls,
                    ..
                } => {
                    let mut parts: Vec<Part> = Vec::new();
                    if let Some(t) = content
                        && !t.is_empty()
                    {
                        parts.push(Part::Text(t));
                    }
                    for tc in tool_calls {
                        let args = serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);
                        parts.push(Part::FunctionCall(FunctionCall {
                            name: tc.name,
                            args,
                        }));
                    }
                    if !parts.is_empty() {
                        contents.push(Content {
                            role: "model",
                            parts,
                        });
                    }
                }
                Message::ToolResult { call_id, content } => {
                    // Gemini matches function responses by name; we store the function
                    // name in call_id (see executor: Gemini tool call id == function name).
                    pending_fn_responses.push(Part::FunctionResponse(FunctionResponse {
                        name: call_id,
                        response: Value::String(content),
                    }));
                }
            }
        }

        if !pending_fn_responses.is_empty() {
            contents.push(Content {
                role: "user",
                parts: pending_fn_responses,
            });
        }

        let tools = req.tools.map(|ts| {
            vec![GeminiTools {
                function_declarations: ts
                    .into_iter()
                    .map(|t: ToolDefinition| FunctionDeclaration {
                        name: t.function.name,
                        description: t.function.description,
                        parameters: t.function.parameters,
                    })
                    .collect(),
            }]
        });

        let tool_config = req.tool_choice.map(|tc| ToolConfig {
            function_calling_config: FunctionCallingConfig {
                mode: match tc {
                    ToolChoice::None => "NONE",
                    ToolChoice::Auto => "AUTO",
                    ToolChoice::Required => "ANY",
                    ToolChoice::Tool(_) => "ANY",
                },
                allowed_function_names: match tc {
                    ToolChoice::Tool(name) => Some(vec![name]),
                    _ => None,
                },
            },
        });

        let generation_config = {
            let gc = GenerationConfig {
                temperature: req.temperature,
                max_output_tokens: req.max_tokens,
                response_mime_type: req.response_format.map(|_| "application/json"),
            };
            if gc.temperature.is_none()
                && gc.max_output_tokens.is_none()
                && gc.response_mime_type.is_none()
            {
                None
            } else {
                Some(gc)
            }
        };

        Request {
            contents,
            system_instruction,
            tools,
            tool_config,
            generation_config,
        }
    }
}

// ── Response ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Response {
    pub candidates: Option<Vec<Candidate>>,
    pub usage_metadata: Option<UsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub content: ResponseContent,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ResponseContent {
    pub parts: Vec<ResponsePart>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResponsePart {
    #[serde(default)]
    pub text: Option<String>,
    pub function_call: Option<ResponseFunctionCall>,
}

#[derive(Debug, Deserialize)]
pub struct ResponseFunctionCall {
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    pub prompt_token_count: u32,
    pub candidates_token_count: u32,
    pub total_token_count: u32,
}

impl From<UsageMetadata> for crate::types::UsageStats {
    fn from(u: UsageMetadata) -> Self {
        Self {
            prompt_tokens: u.prompt_token_count as usize,
            completion_tokens: u.candidates_token_count as usize,
            total_tokens: u.total_token_count as usize,
        }
    }
}

// ── Streaming ─────────────────────────────────────────────────────────────────
// Gemini streaming returns the same Response structure as non-streaming,
// one JSON object per SSE data event.
pub type StreamChunk = Response;

// ── ProviderProtocol ─────────────────────────────────────────────────────────

use crate::markers::Gemini;
use crate::request::ToolCall as AgentToolCall;
use crate::types::{AgentEvent, PartialToolCall, ProviderProtocol, StreamBufs, ToolCallChunk};

impl ProviderProtocol for Gemini {
    type RawRequest = Request;
    type RawResponse = Response;
    type RawChunk = StreamChunk; // = Response

    fn build_raw(req: crate::request::Request) -> Request {
        Request::from(req)
    }
    fn default_base_url() -> &'static str {
        "https://generativelanguage.googleapis.com/v1beta"
    }

    fn parse_response(raw: Response) -> (Vec<AgentEvent>, Vec<AgentToolCall>) {
        let mut events = Vec::new();
        if let Some(u) = raw.usage_metadata {
            events.push(AgentEvent::Usage(u.into()));
        }

        let candidates = match raw.candidates {
            Some(c) => c,
            None => return (events, vec![]),
        };
        let candidate = match candidates.into_iter().next() {
            Some(c) => c,
            None => return (events, vec![]),
        };

        let mut tool_calls = Vec::new();
        for part in candidate.content.parts {
            if let Some(t) = part.text.filter(|s| !s.is_empty()) {
                events.push(AgentEvent::Token(t));
            }
            if let Some(fc) = part.function_call {
                tool_calls.push(AgentToolCall {
                    id: fc.name.clone(),
                    name: fc.name,
                    arguments: serde_json::to_string(&fc.args).unwrap_or_default(),
                });
            }
        }
        (events, tool_calls)
    }

    fn parse_chunk(chunk: StreamChunk, bufs: &mut StreamBufs) -> Vec<AgentEvent> {
        let mut events = Vec::new();
        if let Some(u) = chunk.usage_metadata {
            events.push(AgentEvent::Usage(u.into()));
        }

        let candidates = match chunk.candidates {
            Some(c) => c,
            None => return events,
        };
        let candidate = match candidates.into_iter().next() {
            Some(c) => c,
            None => return events,
        };

        for part in candidate.content.parts {
            if let Some(t) = part.text.filter(|s| !s.is_empty()) {
                bufs.content_buf.push_str(&t);
                events.push(AgentEvent::Token(t));
            }
            if let Some(fc) = part.function_call {
                let idx = bufs.tool_call_bufs.len();
                let args = serde_json::to_string(&fc.args).unwrap_or_default();
                events.push(AgentEvent::ToolCall(ToolCallChunk {
                    id: fc.name.clone(),
                    name: fc.name.clone(),
                    delta: String::new(),
                    index: idx as u32,
                }));
                events.push(AgentEvent::ToolCall(ToolCallChunk {
                    id: fc.name.clone(),
                    name: fc.name.clone(),
                    delta: args.clone(),
                    index: idx as u32,
                }));
                bufs.tool_call_bufs.push(Some(PartialToolCall {
                    id: fc.name.clone(),
                    name: fc.name,
                    arguments: args,
                }));
            }
        }
        events
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

    fn url_suffix(model: &str, streaming: bool) -> String {
        if streaming {
            format!("/models/{}:streamGenerateContent?alt=sse", model)
        } else {
            format!("/models/{}:generateContent", model)
        }
    }

    fn uses_query_key_auth() -> bool {
        true
    }
}
