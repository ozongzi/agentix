pub mod request;
pub mod response;

use eventsource_stream::Eventsource;
use futures::{StreamExt, stream::BoxStream};
use tracing::debug;

use crate::config::AgentConfig;
use crate::error::ApiError;
use crate::msg::LlmEvent;
use crate::provider::{PostConfig, post_json, post_streaming};
use crate::raw::shared::ToolDefinition;
use crate::request::{Message, ToolCall};
use crate::types::{
    CompleteResponse, FinishReason, PartialToolCall, StreamBufs, ToolCallChunk, UsageStats,
};

use response::Response;

pub(crate) async fn stream_gemini(
    token: &str,
    http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
    let req = request::build_gemini_request(config, messages, tools);
    let url = format!(
        "{}/models/{}:streamGenerateContent?alt=sse",
        config.base_url.trim_end_matches('/'),
        config.model,
    );

    let resp = post_streaming(
        http,
        &url,
        &req,
        token,
        &PostConfig {
            use_query_key: true,
            auth_header: None,
            extra_headers: &[],
            max_retries: config.max_retries,
            retry_delay_ms: config.retry_delay_ms,
        },
    )
    .await?;

    Ok(async_stream::stream! {
        let mut bufs = StreamBufs::new();
        let mut sse  = resp.bytes_stream().eventsource();
        let mut saw_finish_reason = false;

        while let Some(ev_res) = sse.next().await {
            match ev_res {
                Ok(ev) => {
                    #[cfg(feature = "sensitive-logs")]
                    if crate::sensitive_logs_enabled() {
                        tracing::info!(body = %ev.data, "received raw streaming response chunk");
                    }
                    match serde_json::from_str::<Response>(&ev.data) {
                        Ok(chunk) => {
                            if chunk
                                .candidates
                                .as_ref()
                                .is_some_and(|candidates| candidates.iter().any(|candidate| candidate.finish_reason.is_some()))
                            {
                                saw_finish_reason = true;
                            }
                            for lev in parse_chunk(chunk, &mut bufs) { yield lev; }
                        },
                        Err(e)    => debug!(data = %ev.data, error = %e, "gemini chunk parse failed"),
                    }
                }
                Err(e) => { yield LlmEvent::Error(e.to_string()); break; }
            }
        }
        if !saw_finish_reason {
            yield LlmEvent::Error("stream ended without finish_reason".to_string());
        }
        for tc in finalize(&mut bufs) { yield LlmEvent::ToolCall(tc); }
        yield LlmEvent::Done;
    }
    .boxed())
}

pub(crate) async fn complete_gemini(
    token: &str,
    http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<CompleteResponse, ApiError> {
    let req = request::build_gemini_request(config, messages, tools);
    let url = format!(
        "{}/models/{}:generateContent",
        config.base_url.trim_end_matches('/'),
        config.model,
    );

    let body = post_json(
        http,
        &url,
        &req,
        token,
        &PostConfig {
            use_query_key: true,
            auth_header: None,
            extra_headers: &[],
            max_retries: config.max_retries,
            retry_delay_ms: config.retry_delay_ms,
        },
    )
    .await?;

    let raw: Response = serde_json::from_str(&body).map_err(ApiError::Json)?;

    let mut content_buf = String::new();
    let mut tool_calls = Vec::new();
    let mut finish_reason = None;

    if let Some(candidate) = raw.candidates.and_then(|mut c| {
        if c.is_empty() {
            None
        } else {
            Some(c.remove(0))
        }
    }) {
        finish_reason = candidate.finish_reason.as_deref().map(FinishReason::from);
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
        content: if content_buf.is_empty() {
            None
        } else {
            Some(content_buf)
        },
        reasoning: None,
        tool_calls,
        provider_data: None,
        usage: raw.usage_metadata.map(UsageStats::from).unwrap_or_default(),
        finish_reason: finish_reason.unwrap_or_default(),
    })
}

fn parse_chunk(chunk: Response, bufs: &mut StreamBufs) -> Vec<LlmEvent> {
    let mut events = Vec::new();
    if let Some(u) = chunk.usage_metadata {
        events.push(LlmEvent::Usage(UsageStats::from(u)));
    }
    let candidate = match chunk.candidates.and_then(|mut c| {
        if c.is_empty() {
            None
        } else {
            Some(c.remove(0))
        }
    }) {
        Some(c) => c,
        None => return events,
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
                id: fc.name.clone(),
                name: fc.name.clone(),
                delta: String::new(),
                index: idx as u32,
            }));
            events.push(LlmEvent::ToolCallChunk(ToolCallChunk {
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

fn finalize(bufs: &mut StreamBufs) -> Vec<ToolCall> {
    bufs.tool_call_bufs
        .drain(..)
        .flatten()
        .map(|p| ToolCall {
            id: p.id,
            name: p.name,
            arguments: p.arguments,
        })
        .collect()
}
