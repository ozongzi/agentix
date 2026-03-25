pub mod request;
pub mod response;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::{StreamExt, stream::BoxStream};
use tracing::debug;

use crate::config::AgentConfig;
use crate::error::ApiError;
use crate::msg::LlmEvent;
use crate::provider::{Provider, PostConfig, post_streaming, post_json};
use crate::request::{Message, Request, ToolCall, ToolChoice};
use crate::raw::shared::ToolDefinition;
use crate::types::{CompleteResponse, PartialToolCall, StreamBufs, ToolCallChunk, UsageStats};

use request::Request as GeminiRequest;
use response::Response;

/// Provider for the Google Gemini API.
pub struct GeminiProvider { token: String }

impl GeminiProvider {
    pub fn new(token: impl Into<String>) -> Self { Self { token: token.into() } }
}

#[async_trait]
impl Provider for GeminiProvider {
    async fn stream(
        &self,
        http:     &reqwest::Client,
        config:   &AgentConfig,
        messages: &[Message],
        tools:    &[ToolDefinition],
    ) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
        let tool_choice = if tools.is_empty() { None } else { Some(ToolChoice::Auto) };
        let req = GeminiRequest::from(Request {
            system_message:  config.system_prompt.clone(),
            messages:        messages.to_vec(),
            model:           config.model.clone(),
            tools:           if tools.is_empty() { None } else { Some(tools.to_vec()) },
            tool_choice,
            stream:          true,
            temperature:     config.temperature,
            max_tokens:      config.max_tokens,
            response_format: None,
            extra_body:      if config.extra_body.is_empty() { None } else { Some(config.extra_body.clone()) },
        });

        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            config.base_url.trim_end_matches('/'),
            config.model,
        );
        let token = self.token.clone();

        let resp = post_streaming(http, &url, &req, &token, &PostConfig {
            use_query_key:  true,
            auth_header:    None,
            extra_headers:  &[],
            max_retries:    config.max_retries,
            retry_delay_ms: config.retry_delay_ms,
        }).await?;

        Ok(async_stream::stream! {
            let mut bufs = StreamBufs::new();
            let mut sse  = resp.bytes_stream().eventsource();

            while let Some(ev_res) = sse.next().await {
                match ev_res {
                    Ok(ev) if ev.data == "[DONE]" => break,
                    Ok(ev) => match serde_json::from_str::<Response>(&ev.data) {
                        Ok(chunk) => for lev in parse_chunk(chunk, &mut bufs) { yield lev; },
                        Err(e)    => debug!(data = %ev.data, error = %e, "gemini chunk parse failed"),
                    },
                    Err(e) => { yield LlmEvent::Error(e.to_string()); break; }
                }
            }
            for tc in finalize(&mut bufs) { yield LlmEvent::ToolCall(tc); }
            yield LlmEvent::Done;
        }.boxed())
    }

    async fn complete(
        &self,
        http:     &reqwest::Client,
        config:   &AgentConfig,
        messages: &[Message],
        tools:    &[ToolDefinition],
    ) -> Result<CompleteResponse, ApiError> {
        let tool_choice = if tools.is_empty() { None } else { Some(ToolChoice::Auto) };
        let req = GeminiRequest::from(Request {
            system_message:  config.system_prompt.clone(),
            messages:        messages.to_vec(),
            model:           config.model.clone(),
            tools:           if tools.is_empty() { None } else { Some(tools.to_vec()) },
            tool_choice,
            stream:          false,
            temperature:     config.temperature,
            max_tokens:      config.max_tokens,
            response_format: None,
            extra_body:      if config.extra_body.is_empty() { None } else { Some(config.extra_body.clone()) },
        });

        let url = format!(
            "{}/models/{}:generateContent",
            config.base_url.trim_end_matches('/'),
            config.model,
        );

        let body = post_json(http, &url, &req, &self.token, &PostConfig {
            use_query_key:  true,
            auth_header:    None,
            extra_headers:  &[],
            max_retries:    config.max_retries,
            retry_delay_ms: config.retry_delay_ms,
        }).await?;

        let raw: Response = serde_json::from_str(&body)
            .map_err(ApiError::Json)?;

        let mut content_buf = String::new();
        let mut tool_calls = Vec::new();

        if let Some(candidate) = raw.candidates.and_then(|mut c| if c.is_empty() { None } else { Some(c.remove(0)) }) {
            for part in candidate.content.parts {
                if let Some(t) = part.text.filter(|s| !s.is_empty()) {
                    content_buf.push_str(&t);
                }
                if let Some(fc) = part.function_call {
                    tool_calls.push(ToolCall {
                        id: fc.name.clone(),
                        name: fc.name,
                        arguments: serde_json::to_string(&fc.args).unwrap_or_default(),
                    });
                }
            }
        }

        Ok(CompleteResponse {
            content: if content_buf.is_empty() { None } else { Some(content_buf) },
            reasoning: None,
            tool_calls,
            usage: raw.usage_metadata.map(UsageStats::from).unwrap_or_default(),
        })
    }
}

fn parse_chunk(chunk: Response, bufs: &mut StreamBufs) -> Vec<LlmEvent> {
    let mut events = Vec::new();
    if let Some(u) = chunk.usage_metadata {
        events.push(LlmEvent::Usage(UsageStats::from(u)));
    }
    let candidate = match chunk.candidates.and_then(|mut c| if c.is_empty() { None } else { Some(c.remove(0)) }) {
        Some(c) => c,
        None    => return events,
    };
    for part in candidate.content.parts {
        if let Some(t) = part.text.filter(|s| !s.is_empty()) {
            bufs.content_buf.push_str(&t);
            events.push(LlmEvent::Token(t));
        }
        if let Some(fc) = part.function_call {
            let idx = bufs.tool_call_bufs.len();
            let args = serde_json::to_string(&fc.args).unwrap_or_default();
            events.push(LlmEvent::ToolCallChunk(ToolCallChunk {
                id: fc.name.clone(), name: fc.name.clone(), delta: String::new(), index: idx as u32,
            }));
            events.push(LlmEvent::ToolCallChunk(ToolCallChunk {
                id: fc.name.clone(), name: fc.name.clone(), delta: args.clone(), index: idx as u32,
            }));
            bufs.tool_call_bufs.push(Some(PartialToolCall {
                id: fc.name.clone(), name: fc.name, arguments: args,
            }));
        }
    }
    events
}

fn finalize(bufs: &mut StreamBufs) -> Vec<ToolCall> {
    bufs.tool_call_bufs.drain(..).flatten().map(|p| ToolCall {
        id: p.id, name: p.name, arguments: p.arguments,
    }).collect()
}
