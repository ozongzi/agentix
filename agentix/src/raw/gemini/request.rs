use serde::Serialize;
use serde_json::Value;

use crate::config::AgentConfig;
use crate::raw::shared::ToolDefinition;
use crate::request::{ImageData, Message, UserContent};

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

#[derive(Debug)]
pub enum Part {
    Text(String),
    InlineData(Blob),
    FunctionCall(FunctionCall),
    FunctionResponse(FunctionResponse),
}

impl serde::Serialize for Part {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(None)?;
        match self {
            Part::Text(t)             => { map.serialize_entry("text", t)?; }
            Part::InlineData(b)       => { map.serialize_entry("inline_data", b)?; }
            Part::FunctionCall(fc)    => { map.serialize_entry("function_call", fc)?; }
            Part::FunctionResponse(fr) => { map.serialize_entry("function_response", fr)?; }
        }
        map.end()
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Blob {
    pub mime_type: String,
    pub data: String,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<serde_json::Value>,
}

pub(crate) fn build_gemini_request(
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Request {
    let system_instruction = config.system_prompt.as_ref()
        .filter(|s| !s.is_empty())
        .map(|s| SystemInstruction { parts: vec![Part::Text(s.clone())] });

    let mut contents: Vec<Content> = Vec::new();
    let mut pending_fn_responses: Vec<Part> = Vec::new();

    for msg in messages {
        match msg {
            Message::User(parts) => {
                if !pending_fn_responses.is_empty() {
                    contents.push(Content {
                        role: "user",
                        parts: std::mem::take(&mut pending_fn_responses),
                    });
                }
                contents.push(Content {
                    role: "user",
                    parts: parts.iter().map(|p| match p {
                        UserContent::Text { text: t } => Part::Text(t.clone()),
                        UserContent::Image(img) => Part::InlineData(Blob {
                            mime_type: img.mime_type.clone(),
                            data: match &img.data {
                                ImageData::Base64(b) => b.clone(),
                                ImageData::Url(u)    => u.clone(),
                            },
                        }),
                    }).collect(),
                });
            }
            Message::Assistant { content, tool_calls, .. } => {
                let mut parts: Vec<Part> = Vec::new();
                if let Some(t) = content && !t.is_empty() {
                    parts.push(Part::Text(t.clone()));
                }
                for tc in tool_calls {
                    let args = serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);
                    parts.push(Part::FunctionCall(FunctionCall { name: tc.name.clone(), args }));
                }
                if !parts.is_empty() {
                    contents.push(Content { role: "model", parts });
                }
            }
            Message::ToolResult { call_id, content } => {
                pending_fn_responses.push(Part::FunctionResponse(FunctionResponse {
                    name: call_id.clone(),
                    response: Value::String(content.clone()),
                }));
            }
        }
    }
    if !pending_fn_responses.is_empty() {
        contents.push(Content { role: "user", parts: pending_fn_responses });
    }

    let gemini_tools = if tools.is_empty() {
        None
    } else {
        Some(vec![GeminiTools {
            function_declarations: tools.iter().map(|t| FunctionDeclaration {
                name: t.function.name.clone(),
                description: t.function.description.clone(),
                parameters: t.function.parameters.clone(),
            }).collect(),
        }])
    };

    let tool_config = if tools.is_empty() {
        None
    } else {
        Some(ToolConfig {
            function_calling_config: FunctionCallingConfig {
                mode: "AUTO",
                allowed_function_names: None,
            },
        })
    };

    let (response_mime_type, response_schema) = match &config.response_format {
        Some(crate::request::ResponseFormat::JsonObject) =>
            (Some("application/json"), None),
        Some(crate::request::ResponseFormat::JsonSchema { schema, .. }) =>
            (Some("application/json"), Some(schema.clone())),
        _ => (None, None),
    };
    let gc = GenerationConfig {
        temperature:        config.temperature,
        max_output_tokens:  config.max_tokens,
        response_mime_type,
        response_schema,
    };
    let generation_config = if gc.temperature.is_none() && gc.max_output_tokens.is_none()
        && gc.response_mime_type.is_none() && gc.response_schema.is_none() {
        None
    } else {
        Some(gc)
    };

    Request { contents, system_instruction, tools: gemini_tools, tool_config, generation_config }
}
