//! Unified request layer.
//!
//! `AgentRequest` is the provider-agnostic representation of a chat completion
//! request. Each provider implements `From<AgentRequest>` (or
//! `Into<ProviderRequest>`) in its own module so that the agent core never
//! needs to know about provider-specific field names or structural quirks.

use serde::{Deserialize, Serialize};
use crate::raw::shared::ToolDefinition;

// ─── Message ────────────────────────────────────────────────────────────────

/// Image content that can be embedded in a user message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageContent {
    pub data: ImageData,
    /// MIME type, e.g. `"image/jpeg"`, `"image/png"`.
    pub mime_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageData {
    Base64(String),
    Url(String),
}

/// A single content block inside a `Message::User`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserContent {
    Text(String),
    Image(ImageContent),
}

impl From<&str> for UserContent {
    fn from(s: &str) -> Self {
        UserContent::Text(s.to_string())
    }
}
impl From<String> for UserContent {
    fn from(s: String) -> Self {
        UserContent::Text(s)
    }
}

/// A single turn in a conversation. Every variant carries exactly the fields
/// it needs — no invalid states are representable.
#[derive(Debug, Clone)]
pub enum Message {
    /// A message from the human side, supporting text and images.
    User(Vec<UserContent>),

    /// A message produced by the model. `content` and `tool_calls` may both be
    /// present; `content` may be absent when the model only emits tool calls.
    Assistant {
        content: Option<String>,
        /// Provider-specific chain-of-thought / reasoning text, if any.
        reasoning: Option<String>,
        tool_calls: Vec<ToolCall>,
    },

    /// The result of a tool invocation, keyed by the call's ID.
    ToolResult { call_id: String, content: String },
}

impl Message {
    /// Estimate the number of tokens in this message using tiktoken.
    ///
    /// Note: This is an estimation. Different providers have slightly different
    /// tokenization rules and overheads for message metadata (role, name, etc.).
    pub fn estimate_tokens(&self) -> usize {
        use std::sync::OnceLock;
        static BPE: OnceLock<tiktoken_rs::CoreBPE> = OnceLock::new();
        let bpe = BPE.get_or_init(|| tiktoken_rs::cl100k_base().unwrap());
        let mut tokens = 0;

        match self {
            Message::User(parts) => {
                tokens += 4; // overhead for role
                for part in parts {
                    match part {
                        UserContent::Text(t) => tokens += bpe.encode_with_special_tokens(t).len(),
                        UserContent::Image(_) => tokens += 1000, // rough fixed cost for images
                    }
                }
            }
            Message::Assistant { content, reasoning, tool_calls } => {
                tokens += 4;
                if let Some(c) = content { tokens += bpe.encode_with_special_tokens(c).len(); }
                if let Some(r) = reasoning { tokens += bpe.encode_with_special_tokens(r).len(); }
                for tc in tool_calls {
                    tokens += bpe.encode_with_special_tokens(&tc.name).len();
                    tokens += bpe.encode_with_special_tokens(&tc.arguments).len();
                }
            }
            Message::ToolResult { content, .. } => {
                tokens += 4;
                tokens += bpe.encode_with_special_tokens(content).len();
            }
        }
        tokens
    }
}

/// A single tool invocation requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Raw JSON string produced by the model.
    pub arguments: String,
}

// ─── ToolChoice ─────────────────────────────────────────────────────────────

/// Provider-agnostic tool selection hint.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// Let the model decide (default when tools are present).
    #[default]
    Auto,
    /// The model must not call any tools.
    None,
    /// The model must call at least one tool.
    Required,
    /// Force a specific tool by name.
    Tool(String),
}

// ─── AgentRequest ───────────────────────────────────────────────────────────

/// The canonical, provider-neutral chat-completion request.
///
/// Fields deliberately omitted (e.g. `top_p`, `logprobs`, `prefix`) are either
/// provider-specific or rarely needed. Pass them through the `extra_body`
/// mechanism on the provider's raw request if you need them.
#[derive(Debug, Clone, Default)]
pub struct Request {
    /// Optional system prompt. `None` means no system message is sent.
    pub system_message: Option<String>,

    /// Conversation history. Must not contain `System` messages — use
    /// `system_message` instead.
    pub messages: Vec<Message>,

    /// Model identifier string (e.g. `"deepseek-chat"`, `"gpt-4o"`).
    pub model: String,

    /// Tools the model may call.
    pub tools: Option<Vec<ToolDefinition>>,

    /// How the model should select tools.
    pub tool_choice: Option<ToolChoice>,

    /// Whether to stream the response. Defaults to `false`.
    pub stream: bool,

    /// Sampling temperature.
    pub temperature: Option<f32>,

    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,

    /// Constrain the output format.
    pub response_format: Option<ResponseFormat>,

    /// Arbitrary extra top-level fields merged into the provider's raw request body.
    /// Use for provider-specific options not modelled here (e.g. `prefix`, `thinking`).
    pub extra_body: Option<serde_json::Map<String, serde_json::Value>>,
}

/// Provider-agnostic output-format hint.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    #[default]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
}
