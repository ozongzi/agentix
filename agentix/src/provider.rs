use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::{StreamExt, stream::BoxStream};
use tracing::{warn, debug};

use crate::config::AgentConfig;
use crate::error::ApiError;
use crate::msg::LlmEvent;
use crate::request::{Message, Request, ToolChoice};
use crate::raw::shared::ToolDefinition;
use crate::types::{ProviderProtocol, StreamBufs, ProtocolEvent};

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
}

// ── Shared HTTP POST helper ────────────────────────────────────────────────────

pub(crate) async fn post_streaming<T: serde::Serialize>(
    http:          &reqwest::Client,
    url:           &str,
    body:          &T,
    token:         &str,
    use_query_key: bool,
    auth_header:   Option<&'static str>,
    extra_headers: &'static [(&'static str, &'static str)],
) -> Result<reqwest::Response, ApiError> {
    let effective_url = if use_query_key {
        if url.contains('?') { format!("{}&key={}", url, token) }
        else                  { format!("{}?key={}", url, token) }
    } else {
        url.to_string()
    };

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

    let resp = builder.send().await.map_err(ApiError::Network)?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_else(|e| e.to_string());
        return Err(ApiError::http(status, body));
    }
    Ok(resp)
}

// ── Generic ProtocolProvider<P> ───────────────────────────────────────────────

pub(crate) struct ProtocolProvider<P> {
    pub(crate) token: String,
    _marker: std::marker::PhantomData<fn() -> P>,
}

impl<P: ProviderProtocol> ProtocolProvider<P> {
    pub(crate) fn new(token: impl Into<String>) -> Self {
        Self { token: token.into(), _marker: std::marker::PhantomData }
    }
}

#[async_trait]
impl<P: ProviderProtocol> Provider for ProtocolProvider<P> {
    async fn stream(
        &self,
        http:     &reqwest::Client,
        config:   &AgentConfig,
        messages: &[Message],
        tools:    &[ToolDefinition],
    ) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
        let tool_choice = if tools.is_empty() { None } else { Some(ToolChoice::Auto) };
        let req = Request {
            system_message:  config.system_prompt.clone(),
            messages:        P::prepare_history(messages.to_vec()),
            model:           config.model.clone(),
            tools:           if tools.is_empty() { None } else { Some(tools.to_vec()) },
            tool_choice,
            stream:          true,
            temperature:     config.temperature,
            max_tokens:      config.max_tokens,
            response_format: None,
            extra_body:      if config.extra_body.is_empty() { None } else { Some(config.extra_body.clone()) },
        };

        let url = format!(
            "{}{}",
            config.base_url.trim_end_matches('/'),
            P::url_suffix(&config.model, true)
        );
        let raw = P::build_raw(req);

        let mut attempts = 0;
        let mut delay = config.retry_delay_ms;

        let resp = loop {
            match post_streaming(
                http, &url, &raw, &self.token,
                P::uses_query_key_auth(),
                P::auth_header_name(),
                P::extra_headers(),
            ).await {
                Ok(r) => break r,
                Err(e) if e.is_retriable() && attempts < config.max_retries => {
                    attempts += 1;
                    warn!(error = %e, attempt = attempts, "transient error, retrying in {}ms", delay);
                    tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                    delay *= 2; 
                }
                Err(e) => return Err(e),
            }
        };

        let out = async_stream::stream! {
            let mut bufs  = StreamBufs::new();
            let mut sse   = resp.bytes_stream().eventsource();

            while let Some(ev_res) = sse.next().await {
                match ev_res {
                    Ok(ev) if ev.data == "[DONE]" => break,
                    Ok(ev) => {
                        match serde_json::from_str::<P::RawChunk>(&ev.data) {
                            Ok(chunk) => {
                                for ae in P::parse_chunk(chunk, &mut bufs) {
                                    match ae {
                                        ProtocolEvent::Token(t)     => yield LlmEvent::Token(t),
                                        ProtocolEvent::Reasoning(t) => yield LlmEvent::Reasoning(t),
                                        ProtocolEvent::ToolCallChunk(tc) => yield LlmEvent::ToolCallChunk(tc),
                                        ProtocolEvent::Usage(stats) => yield LlmEvent::Usage(stats),
                                    }
                                }
                            }
                            Err(e) => {
                                debug!(data = %ev.data, error = %e, "chunk parse failed (ignoring)");
                            }
                        }
                    }
                    Err(e) => {
                        yield LlmEvent::Error(e.to_string());
                        break;
                    }
                }
            }

            for tc in P::finalize_stream(&mut bufs) {
                yield LlmEvent::ToolCall(tc);
            }

            yield LlmEvent::Done;
        };

        Ok(out.boxed())
    }
}

// ── Concrete provider structs ──────────────────────────────────────────────────

macro_rules! make_provider {
    ($name:ident, $marker:ty, $doc:literal) => {
        #[doc = $doc]
        pub struct $name(pub(crate) ProtocolProvider<$marker>);

        impl $name {
            pub fn new(token: impl Into<String>) -> Self {
                Self(ProtocolProvider::new(token))
            }
        }

        #[async_trait]
        impl Provider for $name {
            async fn stream(
                &self,
                http: &reqwest::Client,
                config: &AgentConfig,
                messages: &[Message],
                tools: &[ToolDefinition],
            ) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
                self.0.stream(http, config, messages, tools).await
            }
        }
    };
}

use crate::markers::{Anthropic, DeepSeek, Gemini, OpenAI};

make_provider!(DeepSeekProvider,  DeepSeek,  "Provider for the DeepSeek API.");
make_provider!(OpenAIProvider,    OpenAI,    "Provider for the OpenAI (and compatible) API.");
make_provider!(AnthropicProvider, Anthropic, "Provider for the Anthropic Messages API.");
make_provider!(GeminiProvider,    Gemini,    "Provider for the Google Gemini API.");
