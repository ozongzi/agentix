use async_trait::async_trait;
use futures::stream::BoxStream;
use tracing::warn;

use crate::config::AgentConfig;
use crate::error::ApiError;
use crate::msg::LlmEvent;
use crate::request::Message;
use crate::raw::shared::ToolDefinition;
use crate::types::CompleteResponse;

// ── Provider trait ─────────────────────────────────────────────────────────────

/// Abstracts a single LLM provider.
#[async_trait]
pub trait Provider: Send + Sync {
    async fn stream(
        &self,
        http:     &reqwest::Client,
        config:   &AgentConfig,
        messages: &[Message],
        tools:    &[ToolDefinition],
    ) -> Result<BoxStream<'static, LlmEvent>, ApiError>;

    /// Non-streaming completion. Each provider implements this natively
    /// (i.e. sends `stream: false` and parses the full JSON response).
    async fn complete(
        &self,
        http:     &reqwest::Client,
        config:   &AgentConfig,
        messages: &[Message],
        tools:    &[ToolDefinition],
    ) -> Result<CompleteResponse, ApiError>;
}

// ── Shared HTTP POST helper ────────────────────────────────────────────────────

pub(crate) struct PostConfig {
    pub use_query_key:  bool,
    pub auth_header:    Option<&'static str>,
    pub extra_headers:  &'static [(&'static str, &'static str)],
    pub max_retries:    u32,
    pub retry_delay_ms: u64,
}

pub(crate) async fn post_streaming<T: serde::Serialize>(
    http:   &reqwest::Client,
    url:    &str,
    body:   &T,
    token:  &str,
    cfg:    &PostConfig,
) -> Result<reqwest::Response, ApiError> {
    let use_query_key  = cfg.use_query_key;
    let auth_header    = cfg.auth_header;
    let extra_headers  = cfg.extra_headers;
    let max_retries    = cfg.max_retries;
    let retry_delay_ms = cfg.retry_delay_ms;
    let effective_url = if use_query_key {
        if url.contains('?') { format!("{}&key={}", url, token) }
        else                  { format!("{}?key={}", url, token) }
    } else {
        url.to_string()
    };

    let mut attempts = 0u32;
    let mut delay = retry_delay_ms;
    loop {
        let mut builder = http.post(&effective_url);
        if !use_query_key {
            match auth_header {
                Some(h) => { builder = builder.header(h, token); }
                None    => { builder = builder.bearer_auth(token); }
            }
        }
        for &(name, value) in extra_headers {
            builder = builder.header(name, value);
        }
        builder = builder.json(body);

        match builder.send().await.map_err(ApiError::Network) {
            Ok(resp) if resp.status().is_success() => return Ok(resp),
            Ok(resp) => {
                let status = resp.status();
                let b = resp.text().await.unwrap_or_else(|e| e.to_string());
                let err = ApiError::http(status, b);
                if err.is_retriable() && attempts < max_retries {
                    attempts += 1;
                    warn!(error = %err, attempt = attempts, "transient error, retrying in {}ms", delay);
                    tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                    delay *= 2;
                } else {
                    return Err(err);
                }
            }
            Err(e) => {
                if e.is_retriable() && attempts < max_retries {
                    attempts += 1;
                    warn!(error = %e, attempt = attempts, "transient error, retrying in {}ms", delay);
                    tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                    delay *= 2;
                } else {
                    return Err(e);
                }
            }
        }
    }
}

/// Like `post_streaming`, but expects a full JSON response body (non-streaming).
pub(crate) async fn post_json<T: serde::Serialize>(
    http:   &reqwest::Client,
    url:    &str,
    body:   &T,
    token:  &str,
    cfg:    &PostConfig,
) -> Result<String, ApiError> {
    let resp = post_streaming(http, url, body, token, cfg).await?;
    resp.text().await.map_err(ApiError::Network)
}

// ── Concrete providers (re-exported from raw/) ────────────────────────────────

pub use crate::raw::anthropic::AnthropicProvider;
pub use crate::raw::deepseek::DeepSeekProvider;
pub use crate::raw::openai::OpenAIProvider;
pub use crate::raw::gemini::GeminiProvider;
