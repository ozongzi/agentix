use serde::Serialize;
use serde_json::Value;

use crate::raw::shared::ToolDefinition;
use crate::request::{ImageData, Message, ToolChoice, UserContent};

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
}

impl From<crate::request::Request> for Request {
    fn from(req: crate::request::Request) -> Self {
        let system_instruction = req.system_message
            .filter(|s| !s.is_empty())
            .map(|s| SystemInstruction { parts: vec![Part::Text(s)] });

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
                    contents.push(Content {
                        role: "user",
                        parts: parts.into_iter().map(|p| match p {
                            UserContent::Text(t) => Part::Text(t),
                            UserContent::Image(img) => Part::InlineData(Blob {
                                mime_type: img.mime_type,
                                data: match img.data {
                                    ImageData::Base64(b) => b,
                                    ImageData::Url(u)    => u,
                                },
                            }),
                        }).collect(),
                    });
                }
                Message::Assistant { content, tool_calls, .. } => {
                    let mut parts: Vec<Part> = Vec::new();
                    if let Some(t) = content && !t.is_empty() {
                        parts.push(Part::Text(t));
                    }
                    for tc in tool_calls {
                        let args = serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);
                        parts.push(Part::FunctionCall(FunctionCall { name: tc.name, args }));
                    }
                    if !parts.is_empty() {
                        contents.push(Content { role: "model", parts });
                    }
                }
                Message::ToolResult { call_id, content } => {
                    pending_fn_responses.push(Part::FunctionResponse(FunctionResponse {
                        name: call_id,
                        response: Value::String(content),
                    }));
                }
            }
        }
        if !pending_fn_responses.is_empty() {
            contents.push(Content { role: "user", parts: pending_fn_responses });
        }

        let tools = req.tools.map(|ts| vec![GeminiTools {
            function_declarations: ts.into_iter().map(|t: ToolDefinition| FunctionDeclaration {
                name: t.function.name,
                description: t.function.description,
                parameters: t.function.parameters,
            }).collect(),
        }]);

        let tool_config = req.tool_choice.map(|tc| ToolConfig {
            function_calling_config: FunctionCallingConfig {
                mode: match tc {
                    ToolChoice::None     => "NONE",
                    ToolChoice::Auto     => "AUTO",
                    ToolChoice::Required => "ANY",
                    ToolChoice::Tool(_)  => "ANY",
                },
                allowed_function_names: match tc {
                    ToolChoice::Tool(name) => Some(vec![name]),
                    _                      => None,
                },
            },
        });

        let gc = GenerationConfig {
            temperature:        req.temperature,
            max_output_tokens:  req.max_tokens,
            response_mime_type: req.response_format.map(|_| "application/json"),
        };
        let generation_config = if gc.temperature.is_none() && gc.max_output_tokens.is_none() && gc.response_mime_type.is_none() {
            None
        } else {
            Some(gc)
        };

        Request { contents, system_instruction, tools, tool_config, generation_config }
    }
}
