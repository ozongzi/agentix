use std::sync::{Arc, RwLock};

use futures::stream::BoxStream;

use crate::config::AgentConfig;
use crate::error::ApiError;
use crate::msg::LlmEvent;
use crate::provider::Provider;
use crate::raw::shared::ToolDefinition;
use crate::request::Message;

// ── LlmClient ─────────────────────────────────────────────────────────────────

struct Inner {
    provider: Box<dyn Provider>,
    http:     reqwest::Client,
    config:   RwLock<AgentConfig>,
}

/// A clonable LLM client backed by a [`Provider`].
///
/// All clones share the same HTTP connection pool and configuration —
/// a `config` change on any clone is visible to all others immediately.
///
/// # Configuration
/// Call the setter methods (`model`, `base_url`, etc.) at any time — before
/// or after the first [`Agent`][crate::Agent] is built.  Each change takes
/// effect on the **next** API request.
///
/// # Custom HTTP client
/// Pass your own `reqwest::Client` (with timeouts, proxies, etc.) via
/// [`LlmClient::with_http`].
#[derive(Clone)]
pub struct LlmClient(Arc<Inner>);

impl LlmClient {
    pub fn new(provider: impl Provider + 'static, config: AgentConfig) -> Self {
        Self::with_http(provider, reqwest::Client::new(), config)
    }

    pub fn with_http(
        provider: impl Provider + 'static,
        http:     reqwest::Client,
        config:   AgentConfig,
    ) -> Self {
        Self(Arc::new(Inner {
            provider: Box::new(provider),
            http,
            config: RwLock::new(config),
        }))
    }

    // ── Config setters (interior mutability) ─────────────────────────────────

    pub fn model(&self, m: impl Into<String>) -> &Self {
        self.0.config.write().unwrap().model = m.into();
        self
    }

    pub fn base_url(&self, url: impl Into<String>) -> &Self {
        self.0.config.write().unwrap().base_url = url.into();
        self
    }

    pub fn system_prompt(&self, p: impl Into<String>) -> &Self {
        self.0.config.write().unwrap().system_prompt = Some(p.into());
        self
    }

    pub fn max_tokens(&self, n: u32) -> &Self {
        self.0.config.write().unwrap().max_tokens = Some(n);
        self
    }

    pub fn temperature(&self, t: f32) -> &Self {
        self.0.config.write().unwrap().temperature = Some(t);
        self
    }

    pub fn clear_system_prompt(&self) -> &Self {
        self.0.config.write().unwrap().system_prompt = None;
        self
    }

    // ── Config read ───────────────────────────────────────────────────────────

    pub fn snapshot(&self) -> AgentConfig {
        self.0.config.read().unwrap().clone()
    }

    // ── Stream ────────────────────────────────────────────────────────────────

    pub(crate) async fn stream(
        &self,
        messages: &[Message],
        tools:    &[ToolDefinition],
    ) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
        let config = self.0.config.read().unwrap().clone();
        self.0.provider.stream(&self.0.http, &config, messages, tools).await
    }
}

// ── Convenience constructors ──────────────────────────────────────────────────

impl LlmClient {
    pub fn deepseek(token: impl Into<String>) -> Self {
        use crate::provider::DeepSeekProvider;
        use crate::markers::DeepSeek;
        use crate::types::ProviderProtocol;
        let config = AgentConfig {
            base_url: DeepSeek::default_base_url().to_string(),
            model:    "deepseek-chat".to_string(),
            ..Default::default()
        };
        Self::new(DeepSeekProvider::new(token), config)
    }

    pub fn openai(token: impl Into<String>) -> Self {
        use crate::provider::OpenAIProvider;
        use crate::markers::OpenAI;
        use crate::types::ProviderProtocol;
        let config = AgentConfig {
            base_url: OpenAI::default_base_url().to_string(),
            model:    "gpt-4o".to_string(),
            ..Default::default()
        };
        Self::new(OpenAIProvider::new(token), config)
    }

    pub fn anthropic(token: impl Into<String>) -> Self {
        use crate::provider::AnthropicProvider;
        use crate::markers::Anthropic;
        use crate::types::ProviderProtocol;
        let config = AgentConfig {
            base_url: Anthropic::default_base_url().to_string(),
            model:    "claude-opus-4-5".to_string(),
            ..Default::default()
        };
        Self::new(AnthropicProvider::new(token), config)
    }

    pub fn gemini(token: impl Into<String>) -> Self {
        use crate::provider::GeminiProvider;
        use crate::markers::Gemini;
        use crate::types::ProviderProtocol;
        let config = AgentConfig {
            base_url: Gemini::default_base_url().to_string(),
            model:    "gemini-2.0-flash".to_string(),
            ..Default::default()
        };
        Self::new(GeminiProvider::new(token), config)
    }
}
