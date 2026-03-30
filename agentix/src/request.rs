//! Unified request layer.
//!
//! [`Request`] is the self-contained, provider-agnostic chat-completion request.
//! It carries *everything* needed to hit an LLM API — provider, credentials,
//! model, messages, tools, and tuning knobs.
//!
//! ```no_run
//! use agentix::{Request, Provider, Message, UserContent, LlmEvent};
//! use futures::StreamExt;
//!
//! # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let http = reqwest::Client::new();
//!
//! let mut stream = Request::new(Provider::DeepSeek, "sk-...")
//!     .model("deepseek-chat")
//!     .system_prompt("You are helpful.")
//!     .user("Hello!")
//!     .stream(&http)
//!     .await?;
//!
//! while let Some(event) = stream.next().await {
//!     if let LlmEvent::Token(t) = event { print!("{t}"); }
//! }
//! # Ok(()) }
//! ```

use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::msg::LlmEvent;
use crate::raw::shared::ToolDefinition;
use crate::types::CompleteResponse;

// ─── Message ────────────────────────────────────────────────────────────────

/// Image content that can be embedded in a user message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageContent {
    /// The image payload.
    pub data: ImageData,
    /// MIME type, e.g. `"image/jpeg"`, `"image/png"`.
    pub mime_type: String,
}

/// How the image data is provided.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageData {
    /// Base64-encoded image bytes.
    Base64(String),
    /// Publicly accessible URL.
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

/// Drop the oldest messages from `history` until the total estimated token
/// count is at or below `budget`.  Always keeps at least one message.
pub fn truncate_to_token_budget(history: &mut Vec<Message>, budget: usize) {
    // Scan from the back, accumulating tokens until we exceed the budget.
    // The first index (from the front) where the suffix fits is the cut point.
    let mut acc: usize = 0;
    let mut keep_from = history.len(); // default: keep all
    for (i, msg) in history.iter().enumerate().rev() {
        acc += msg.estimate_tokens();
        if acc > budget {
            // suffix starting at i+1 fits; drop everything before it,
            // but always keep at least one message.
            keep_from = (i + 1).min(history.len() - 1);
            break;
        }
    }
    if keep_from > 0 {
        history.drain(0..keep_from);
    }
}

/// A single tool invocation requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned call ID (used to match results back).
    pub id: String,
    /// Name of the tool the model wants to invoke.
    pub name: String,
    /// Raw JSON string produced by the model.
    pub arguments: String,
}

// ─── Provider ──────────────────────────────────────────────────────────────

/// Which LLM provider to use.
///
/// Each variant determines the request/response format, auth method, and
/// default base URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provider {
    DeepSeek,
    OpenAI,
    Anthropic,
    Gemini,
}

impl Provider {
    /// Default base URL for this provider.
    pub fn default_base_url(&self) -> &'static str {
        match self {
            Provider::DeepSeek  => "https://api.deepseek.com",
            Provider::OpenAI    => "https://api.openai.com/v1",
            Provider::Anthropic => "https://api.anthropic.com",
            Provider::Gemini    => "https://generativelanguage.googleapis.com/v1beta",
        }
    }

    /// Default model for this provider.
    pub fn default_model(&self) -> &'static str {
        match self {
            Provider::DeepSeek  => "deepseek-chat",
            Provider::OpenAI    => "gpt-4o",
            Provider::Anthropic => "claude-sonnet-4-20250514",
            Provider::Gemini    => "gemini-2.0-flash",
        }
    }
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

// ─── Request ────────────────────────────────────────────────────────────────

/// A self-contained, provider-agnostic chat-completion request.
///
/// Carries everything needed to hit an LLM API: provider, credentials, model,
/// messages, tools, and tuning parameters.
///
/// Call [`stream()`][Request::stream] or [`complete()`][Request::complete] with
/// a shared `reqwest::Client` to send the request.
#[derive(Debug, Clone)]
pub struct Request {
    // ── Identity ──────────────────────────────────────────────────────────

    /// Which provider to use.
    pub provider: Provider,
    /// API key / token.
    pub api_key: String,
    /// Base URL override. If empty, uses [`Provider::default_base_url`].
    pub base_url: String,

    // ── Model & messages ─────────────────────────────────────────────────

    /// Model identifier (e.g. `"deepseek-chat"`, `"gpt-4o"`).
    pub model: String,
    /// Optional system prompt.
    pub system_message: Option<String>,
    /// Conversation history.
    pub messages: Vec<Message>,

    // ── Tools ────────────────────────────────────────────────────────────

    /// Tools the model may call.
    pub tools: Vec<ToolDefinition>,
    /// How the model should select tools.
    pub tool_choice: Option<ToolChoice>,

    // ── Tuning ───────────────────────────────────────────────────────────

    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Constrain the output format.
    pub response_format: Option<ResponseFormat>,
    /// Arbitrary extra top-level JSON fields merged into the provider's
    /// raw request body (e.g. `prefix`, `thinking`).
    pub extra_body: serde_json::Map<String, serde_json::Value>,

    // ── Retry ────────────────────────────────────────────────────────────

    /// Maximum retries for transient errors. Default: 3.
    pub max_retries: u32,
    /// Initial retry delay in milliseconds. Default: 1000.
    pub retry_delay_ms: u64,
}

impl Request {
    /// Create a new request for the given provider and API key.
    ///
    /// Sets sensible defaults: provider's default base URL and model,
    /// 3 retries with 1 s initial delay, no system prompt.
    pub fn new(provider: Provider, api_key: impl Into<String>) -> Self {
        Self {
            base_url: provider.default_base_url().to_string(),
            model: provider.default_model().to_string(),
            api_key: api_key.into(),
            provider,
            system_message: None,
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            temperature: None,
            max_tokens: None,
            response_format: None,
            extra_body: serde_json::Map::new(),
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    }

    /// Shortcut for `Request::new(Provider::DeepSeek, api_key)`.
    pub fn deepseek(api_key: impl Into<String>) -> Self { Self::new(Provider::DeepSeek, api_key) }

    /// Shortcut for `Request::new(Provider::OpenAI, api_key)`.
    pub fn openai(api_key: impl Into<String>) -> Self { Self::new(Provider::OpenAI, api_key) }

    /// Shortcut for `Request::new(Provider::Anthropic, api_key)`.
    pub fn anthropic(api_key: impl Into<String>) -> Self { Self::new(Provider::Anthropic, api_key) }

    /// Shortcut for `Request::new(Provider::Gemini, api_key)`.
    pub fn gemini(api_key: impl Into<String>) -> Self { Self::new(Provider::Gemini, api_key) }

    // ── Builder setters (all consume & return Self) ──────────────────────

    /// Override the base URL.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the model.
    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = m.into();
        self
    }

    /// Set the system prompt.
    pub fn system_prompt(mut self, p: impl Into<String>) -> Self {
        self.system_message = Some(p.into());
        self
    }

    /// Append a message to the conversation.
    pub fn message(mut self, m: Message) -> Self {
        self.messages.push(m);
        self
    }

    /// Append a user text message (convenience).
    pub fn user(self, text: impl Into<String>) -> Self {
        self.message(Message::User(vec![UserContent::Text(text.into())]))
    }

    /// Set the full message history.
    pub fn messages(mut self, msgs: Vec<Message>) -> Self {
        self.messages = msgs;
        self
    }

    /// Set the tool definitions.
    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Set the temperature.
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Set the max tokens.
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Set the response format to plain text (the default).
    pub fn text(mut self) -> Self {
        self.response_format = Some(ResponseFormat::Text);
        self
    }

    /// Constrain output to a named JSON Schema (OpenAI `json_schema` mode).
    ///
    /// Use `schemars::schema_for!(T)` to generate the schema:
    /// ```ignore
    /// let schema = serde_json::to_value(schemars::schema_for!(MyStruct)).unwrap();
    /// let req = Request::openai(key).json_schema("my_struct", schema, true);
    /// ```
    pub fn json_schema(mut self, name: impl Into<String>, schema: serde_json::Value, strict: bool) -> Self {
        self.response_format = Some(ResponseFormat::JsonSchema { name: name.into(), schema, strict });
        self
    }

    /// Set the response format to JSON object mode.
    ///
    /// The model will be constrained to emit a valid JSON object. You must
    /// also instruct the model to produce JSON in your system prompt or user
    /// message — the format flag alone is not sufficient for most providers.
    pub fn json(mut self) -> Self {
        self.response_format = Some(ResponseFormat::JsonObject);
        self
    }

    /// Set retry parameters.
    pub fn retries(mut self, max: u32, initial_delay_ms: u64) -> Self {
        self.max_retries = max;
        self.retry_delay_ms = initial_delay_ms;
        self
    }

    /// Merge extra JSON fields into the request body.
    pub fn extra_body(mut self, extra: serde_json::Map<String, serde_json::Value>) -> Self {
        self.extra_body = extra;
        self
    }

    // ── Effective base URL ───────────────────────────────────────────────

    /// Resolve the effective base URL (custom or provider default).
    pub fn effective_base_url(&self) -> &str {
        if self.base_url.is_empty() {
            self.provider.default_base_url()
        } else {
            &self.base_url
        }
    }

    // ── Send ─────────────────────────────────────────────────────────────

    /// Send a streaming request and return a stream of [`LlmEvent`]s.
    pub async fn stream(
        &self,
        http: &reqwest::Client,
    ) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
        let config = self.to_agent_config();
        let messages = &self.messages;
        let tools = &self.tools;

        match self.provider {
            Provider::DeepSeek => {
                use crate::raw::openai::stream_openai_compatible;
                use crate::raw::deepseek::prepare_history;
                let config = degrade_json_schema_for_deepseek(config);
                stream_openai_compatible(
                    &self.api_key, http, &config, messages, tools,
                    Some(prepare_history),
                ).await
            }
            Provider::OpenAI => {
                use crate::raw::openai::stream_openai_compatible;
                stream_openai_compatible(
                    &self.api_key, http, &config, messages, tools, None,
                ).await
            }
            Provider::Anthropic => {
                crate::raw::anthropic::stream_anthropic(
                    &self.api_key, http, &config, messages, tools,
                ).await
            }
            Provider::Gemini => {
                crate::raw::gemini::stream_gemini(
                    &self.api_key, http, &config, messages, tools,
                ).await
            }
        }
    }

    /// Send a non-streaming request and return the complete response.
    pub async fn complete(
        &self,
        http: &reqwest::Client,
    ) -> Result<CompleteResponse, ApiError> {
        let config = self.to_agent_config();
        let messages = &self.messages;
        let tools = &self.tools;

        match self.provider {
            Provider::DeepSeek => {
                use crate::raw::openai::complete_openai_compatible;
                use crate::raw::deepseek::prepare_history;
                let config = degrade_json_schema_for_deepseek(config);
                complete_openai_compatible(
                    &self.api_key, http, &config, messages, tools,
                    Some(prepare_history),
                ).await
            }
            Provider::OpenAI => {
                use crate::raw::openai::complete_openai_compatible;
                complete_openai_compatible(
                    &self.api_key, http, &config, messages, tools, None,
                ).await
            }
            Provider::Anthropic => {
                crate::raw::anthropic::complete_anthropic(
                    &self.api_key, http, &config, messages, tools,
                ).await
            }
            Provider::Gemini => {
                crate::raw::gemini::complete_gemini(
                    &self.api_key, http, &config, messages, tools,
                ).await
            }
        }
    }

    /// Convert to the legacy `AgentConfig` for internal provider use.

    /// This is a temporary bridge until providers are fully migrated.
    fn to_agent_config(&self) -> crate::config::AgentConfig {
        crate::config::AgentConfig {
            base_url: self.effective_base_url().to_string(),
            model: self.model.clone(),
            system_prompt: self.system_message.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            extra_body: self.extra_body.clone(),
            response_format: self.response_format.clone(),
            max_retries: self.max_retries,
            retry_delay_ms: self.retry_delay_ms,
        }
    }
}

/// DeepSeek supports `json_object` but not `json_schema`.
/// Silently degrade so callers don't have to branch on provider.
fn degrade_json_schema_for_deepseek(mut config: crate::config::AgentConfig) -> crate::config::AgentConfig {
    if matches!(config.response_format, Some(ResponseFormat::JsonSchema { .. })) {
        tracing::warn!("DeepSeek does not support json_schema; degrading to json_object");
        config.response_format = Some(ResponseFormat::JsonObject);
    }
    config
}

/// Provider-agnostic output-format hint.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    #[default]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    /// Strict JSON Schema output (OpenAI `json_schema` mode).
    /// `schema` should be a `schemars::schema_for!(T)` value serialized to `Value`.
    JsonSchema {
        /// Name for the schema (shown in API responses).
        name: String,
        /// The JSON Schema object.
        schema: serde_json::Value,
        /// Whether to enforce strict schema adherence.
        strict: bool,
    },
}
