use std::marker::PhantomData;

use crate::api::ApiClient;
use crate::request::{ResponseFormat, Message};
use crate::summarizer::{LlmSummarizer, Summarizer};
use crate::tool_trait::{Tool, ToolBundle};
use tokio::sync::mpsc;

// ── Provider marker types ─────────────────────────────────────────────────────

/// Marker for the DeepSeek API (uses DeepSeek-specific fields like `reasoning_content`, `prefix`).
pub struct DeepSeek;

/// Marker for the standard OpenAI chat completions wire format.
/// Covers GPT models, most OpenRouter routes, local llama.cpp, etc.
pub struct OpenAI;

/// Marker for the Anthropic Messages API wire format.
pub struct Anthropic;

/// Marker for the Google Gemini API wire format.
pub struct Gemini;

// ── Type aliases ──────────────────────────────────────────────────────────────

/// Agent targeting the DeepSeek API.
pub type DeepSeekAgent = Agent<DeepSeek>;

/// Agent targeting any standard OpenAI-compatible endpoint.
pub type OpenAIAgent = Agent<OpenAI>;

/// Agent targeting the Anthropic Messages API.
pub type AnthropicAgent = Agent<Anthropic>;

/// Agent targeting the Google Gemini API.
pub type GeminiAgent = Agent<Gemini>;

// ── Re-export shared types ───────────────────────────────────────────────────
pub use crate::types::{AgentEvent, PartialToolCall, ProviderProtocol, StreamBufs, ToolCallChunk, ToolCallResult};

/// A runtime tool-injection command sent through the channel created by
/// [`Agent::tool_inject_sender`].
pub enum ToolCommand {
    Add(Box<dyn Tool>),
    Remove(Vec<String>),
}

// ── Agent<P> ─────────────────────────────────────────────────────────────────

/// A generic agent parameterised by provider `P`.
///
/// Use the type aliases [`DeepSeekAgent`], [`AnthropicAgent`], [`GeminiAgent`]
/// for the most common cases, or spell out `Agent<MyProvider>` for custom ones.
pub struct Agent<P> {
    pub(crate) client: ApiClient<P>,
    pub(crate) history: Vec<Message>,
    pub(crate) system_prompt: Option<String>,
    pub(crate) summarizer: Box<dyn Summarizer + Send + Sync>,
    pub(crate) auto_summary: bool,
    pub(crate) tool_bundle: ToolBundle,
    pub(crate) streaming: bool,
    pub(crate) model: String,
    pub(crate) temperature: Option<f32>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) response_format: Option<ResponseFormat>,
    pub(crate) interrupt_tx: mpsc::UnboundedSender<String>,
    pub(crate) interrupt_rx: mpsc::UnboundedReceiver<String>,
    pub(crate) tool_inject_tx: mpsc::UnboundedSender<ToolCommand>,
    pub(crate) tool_inject_rx: mpsc::UnboundedReceiver<ToolCommand>,
    pub(crate) extra_body: Option<serde_json::Map<String, serde_json::Value>>,
    _provider: PhantomData<P>,
}

// ── Shared constructor (crate-private) ───────────────────────────────────────

impl<P: ProviderProtocol> Agent<P> {
    pub(crate) fn from_parts(client: ApiClient<P>, model: impl Into<String>) -> Self {
        let (interrupt_tx, interrupt_rx) = mpsc::unbounded_channel();
        let (tool_inject_tx, tool_inject_rx) = mpsc::unbounded_channel();
        let model: String = model.into();
        Self {
            summarizer: Box::new(LlmSummarizer::new(client.clone(), &model)),
            auto_summary: true,
            client,
            history: vec![],
            system_prompt: None,
            tool_bundle: ToolBundle::new(),
            streaming: false,
            model,
            temperature: None,
            max_tokens: None,
            response_format: None,
            interrupt_tx,
            interrupt_rx,
            tool_inject_tx,
            tool_inject_rx,
            extra_body: None,
            _provider: PhantomData,
        }
    }
}

// ── Provider-specific constructors ───────────────────────────────────────────

impl Agent<DeepSeek> {
    /// Create a new agent targeting the DeepSeek API with `deepseek-chat`.
    pub fn new(token: impl Into<String>) -> Self {
        Self::from_parts(ApiClient::new(token), "deepseek-chat")
    }

    /// Create an agent targeting a DeepSeek-compatible or OpenAI-compatible
    /// endpoint with an explicit base URL and model.
    pub fn custom(
        token: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self::from_parts(ApiClient::new(token).with_base_url(base_url), model)
    }
}

impl Agent<OpenAI> {
    /// Create an agent targeting any OpenAI-compatible endpoint.
    pub fn new(
        token: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self::from_parts(ApiClient::new(token).with_base_url(base_url), model)
    }

    /// Create an agent targeting the official OpenAI API (`https://api.openai.com/v1`).
    pub fn official(token: impl Into<String>, model: impl Into<String>) -> Self {
        Self::from_parts(ApiClient::new(token), model)
    }
}

impl Agent<Anthropic> {
    /// Create an agent targeting any Anthropic-compatible endpoint.
    pub fn new(
        token: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self::from_parts(ApiClient::new(token).with_base_url(base_url), model)
    }

    /// Create an agent targeting the official Anthropic API (`https://api.anthropic.com`).
    pub fn official(token: impl Into<String>, model: impl Into<String>) -> Self {
        Self::from_parts(ApiClient::new(token), model)
    }
}

impl Agent<Gemini> {
    /// Create an agent targeting any Gemini-compatible endpoint.
    pub fn new(
        token: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self::from_parts(ApiClient::new(token).with_base_url(base_url), model)
    }

    /// Create an agent targeting the official Google Gemini API
    /// (`https://generativelanguage.googleapis.com/v1beta`).
    pub fn official(token: impl Into<String>, model: impl Into<String>) -> Self {
        Self::from_parts(ApiClient::new(token), model)
    }
}

// ── Shared builder methods ────────────────────────────────────────────────────

impl<P: ProviderProtocol> Agent<P> {
    pub fn with_tool<TT: Tool + 'static>(mut self, tool: TT) -> Self {
        self.tool_bundle = self.tool_bundle.with(tool);
        self
    }

    pub fn chat(mut self, user_message: &str) -> crate::agent::stream::AgentStream<P> {
        self.history.push(Message::User(vec![
            crate::request::UserContent::Text(user_message.to_string()),
        ]));
        crate::agent::stream::AgentStream::new(self)
    }

    pub fn chat_from_history(self) -> crate::agent::stream::AgentStream<P> {
        crate::agent::stream::AgentStream::new(self)
    }

    pub fn streaming(mut self) -> Self {
        self.streaming = true;
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn extra_body(mut self, map: serde_json::Map<String, serde_json::Value>) -> Self {
        if let Some(ref mut existing) = self.extra_body {
            existing.extend(map);
        } else {
            self.extra_body = Some(map);
        }
        self
    }

    pub fn extra_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        if let Some(ref mut m) = self.extra_body {
            m.insert(key.into(), value);
        } else {
            let mut m = serde_json::Map::new();
            m.insert(key.into(), value);
            self.extra_body = Some(m);
        }
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Attach a summarizer and enable automatic summarization.
    pub fn with_summarizer(mut self, summarizer: impl Summarizer + 'static) -> Self {
        self.summarizer = Box::new(summarizer);
        self.auto_summary = true;
        self
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    pub fn with_response_format(mut self, f: ResponseFormat) -> Self {
        self.response_format = Some(f);
        self
    }

    /// Seed the agent with an existing message history.
    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        self.history = history;
        self
    }

    /// Append a user message to history.
    pub fn push_user_message(&mut self, text: &str) {
        self.history.push(Message::User(vec![
            crate::request::UserContent::Text(text.to_string()),
        ]));
    }

    /// Read-only view of the current history.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    pub fn interrupt_sender(&self) -> mpsc::UnboundedSender<String> {
        self.interrupt_tx.clone()
    }

    pub fn tool_inject_sender(&self) -> mpsc::UnboundedSender<ToolCommand> {
        self.tool_inject_tx.clone()
    }

    pub(crate) fn drain_interrupts(&mut self) {
        while let Ok(msg) = self.interrupt_rx.try_recv() {
            self.history.push(Message::User(vec![
                crate::request::UserContent::Text(msg),
            ]));
        }
    }

    pub(crate) fn drain_tool_injections(&mut self) {
        while let Ok(injection) = self.tool_inject_rx.try_recv() {
            match injection {
                ToolCommand::Add(tool) => {
                    self.tool_bundle.push_boxed(tool);
                }
                ToolCommand::Remove(names) => {
                    self.tool_bundle.remove_by_names(&names);
                }
            }
        }
    }
}
