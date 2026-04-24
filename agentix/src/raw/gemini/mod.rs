//! Gemini provider — supports `generationConfig.thinkingConfig` and
//! `thoughtSignature` round-trip.
//!
//! # Reasoning semantics
//!
//! `LlmEvent::Reasoning(text)` streams Gemini's summarized thought parts
//! (`thought: true`). The authoritative state needed for multi-turn tool
//! loops is the `thoughtSignature` that sits on a small number of parts
//! (Gemini 3 requires it on the first `functionCall` per step or it 400s).
//! That state round-trips via [`LlmEvent::AssistantState`] →
//! [`Message::Assistant::provider_data`] (envelope tag `gemini_parts`).

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
        // Accumulate raw part JSONs across chunks so we can emit them verbatim
        // (with thoughtSignature) via provider_data at end of turn.
        let mut raw_parts: Vec<serde_json::Value> = Vec::new();
        let mut sse = resp.bytes_stream().eventsource();
        let mut saw_finish_reason = false;

        while let Some(ev_res) = sse.next().await {
            match ev_res {
                Ok(ev) => {
                    #[cfg(feature = "sensitive-logs")]
                    if crate::sensitive_logs_enabled() {
                        tracing::info!(body = %ev.data, "received raw streaming response chunk");
                    }

                    // Extract raw parts for round-trip before the typed parse
                    // consumes the buffer.
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&ev.data)
                        && let Some(parts) = v
                            .get("candidates")
                            .and_then(|c| c.get(0))
                            .and_then(|c| c.get("content"))
                            .and_then(|c| c.get("parts"))
                            .and_then(|p| p.as_array())
                    {
                        for p in parts { raw_parts.push(p.clone()); }
                    }

                    match serde_json::from_str::<Response>(&ev.data) {
                        Ok(chunk) => {
                            if chunk
                                .candidates
                                .as_ref()
                                .is_some_and(|candidates| candidates.iter().any(|c| c.finish_reason.is_some()))
                            {
                                saw_finish_reason = true;
                            }
                            for lev in parse_chunk(chunk, &mut bufs) { yield lev; }
                        }
                        Err(e) => debug!(data = %ev.data, error = %e, "gemini chunk parse failed"),
                    }
                }
                Err(e) => { yield LlmEvent::Error(e.to_string()); break; }
            }
        }
        if !saw_finish_reason {
            yield LlmEvent::Error("stream ended without finish_reason".to_string());
        }
        for tc in finalize(&mut bufs) { yield LlmEvent::ToolCall(tc); }

        // Gate: only emit AssistantState when the turn has thinking AND
        // function_call. Pure-reasoning turns don't require signature
        // round-trip on any Gemini model.
        if should_emit_state(&raw_parts) {
            yield LlmEvent::AssistantState(serde_json::json!({
                "gemini_parts": raw_parts,
            }));
        }

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

    // Parse twice — typed for extraction, raw Value to preserve part objects
    // with thoughtSignature for next-turn round-trip.
    let raw_value: serde_json::Value = serde_json::from_str(&body).map_err(ApiError::Json)?;
    let raw: Response = serde_json::from_str(&body).map_err(ApiError::Json)?;

    let mut content_buf = String::new();
    let mut reasoning_buf = String::new();
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
                if part.thought == Some(true) {
                    reasoning_buf.push_str(&t);
                } else {
                    content_buf.push_str(&t);
                }
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

    // Grab the raw parts array for round-trip (only if gate holds).
    let raw_parts: Option<Vec<serde_json::Value>> = raw_value
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array())
        .cloned();

    let provider_data = raw_parts
        .as_ref()
        .filter(|parts| should_emit_state(parts))
        .map(|parts| {
            serde_json::json!({
                "gemini_parts": parts,
            })
        });

    Ok(CompleteResponse {
        content: if content_buf.is_empty() {
            None
        } else {
            Some(content_buf)
        },
        reasoning: if reasoning_buf.is_empty() {
            None
        } else {
            Some(reasoning_buf)
        },
        tool_calls,
        provider_data,
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
            if part.thought == Some(true) {
                bufs.reasoning_buf.push_str(&t);
                events.push(LlmEvent::Reasoning(t));
            } else {
                bufs.content_buf.push_str(&t);
                events.push(LlmEvent::Token(t));
            }
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

/// Gate for emitting `provider_data` — only when the turn contains both a
/// thinking-ish artifact (a `thought:true` part OR a `thoughtSignature`) AND
/// at least one `functionCall`. Pure reasoning→text turns are terminal.
fn should_emit_state(parts: &[serde_json::Value]) -> bool {
    let mut has_thinking = false;
    let mut has_function_call = false;
    for p in parts {
        if p.get("thought").and_then(|v| v.as_bool()).unwrap_or(false) {
            has_thinking = true;
        }
        if p.get("thoughtSignature").is_some() {
            has_thinking = true;
        }
        if p.get("functionCall").is_some() {
            has_function_call = true;
        }
    }
    has_thinking && has_function_call
}
