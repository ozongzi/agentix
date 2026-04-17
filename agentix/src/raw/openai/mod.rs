pub mod request;
pub mod response;

use eventsource_stream::Eventsource;
use futures::{StreamExt, stream::BoxStream};
use tracing::debug;

use crate::config::AgentConfig;
use crate::error::ApiError;
use crate::msg::LlmEvent;
use crate::provider::{PostConfig, post_streaming, post_json};
use crate::request::{Message, ToolCall, ToolChoice};
use crate::raw::shared::ToolDefinition;
use crate::types::{CompleteResponse, FinishReason, PartialToolCall, StreamBufs, ToolCallChunk, UsageStats};

use response::{StreamChunk, DeltaToolCall};

pub(crate) async fn stream_openai_compatible(
    token:       &str,
    http:        &reqwest::Client,
    config:      &AgentConfig,
    messages:    &[Message],
    tools:       &[ToolDefinition],
    prepare_history: Option<fn(Vec<Message>) -> Vec<Message>>,
) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
    let tool_choice = if tools.is_empty() { None } else { Some(ToolChoice::Auto) };
    let history = match prepare_history {
        Some(f) => f(messages.to_vec()),
        None    => messages.to_vec(),
    };
    let req = request::build_oai_request(
        config, history, tools, tool_choice, true,
    );
    let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));
    let token = token.to_string();
    let resp = post_streaming(http, &url, &req, &token, &PostConfig {
        use_query_key:  false,
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
                Ok(ev) => match serde_json::from_str::<StreamChunk>(&ev.data) {
                    Ok(chunk) => for lev in parse_chunk(chunk, &mut bufs) { yield lev; },
                    Err(e)    => debug!(data = %ev.data, error = %e, "openai chunk parse failed"),
                },
                Err(e) => { yield LlmEvent::Error(e.to_string()); break; }
            }
        }
        for tc in finalize(&mut bufs) { yield LlmEvent::ToolCall(tc); }
        yield LlmEvent::Done;
    }.boxed())
}

fn parse_chunk(chunk: StreamChunk, bufs: &mut StreamBufs) -> Vec<LlmEvent> {
    let mut events = Vec::new();
    if let Some(u) = chunk.usage {
        events.push(LlmEvent::Usage(UsageStats::from(u)));
    }
    let choice = match chunk.choices.into_iter().next() {
        Some(c) => c,
        None    => return events,
    };
    let delta = choice.delta;
    if let Some(dtcs) = delta.tool_calls {
        for dtc in dtcs {
            events.extend(handle_tool_call_delta(dtc, bufs));
        }
    }
    if let Some(t) = delta.content.filter(|s| !s.is_empty()) {
        bufs.content_buf.push_str(&t);
        events.push(LlmEvent::Token(t));
    }
    events
}

fn handle_tool_call_delta(dtc: DeltaToolCall, bufs: &mut StreamBufs) -> Vec<LlmEvent> {
    let mut events = Vec::new();
    let idx = dtc.index as usize;
    if bufs.tool_call_bufs.len() <= idx {
        bufs.tool_call_bufs.resize_with(idx + 1, || None);
    }
    let entry = &mut bufs.tool_call_bufs[idx];
    if entry.is_none() {
        let id   = dtc.id.clone().unwrap_or_default();
        let name = dtc.function.as_ref().and_then(|f| f.name.clone()).unwrap_or_default();
        events.push(LlmEvent::ToolCallChunk(ToolCallChunk {
            id: id.clone(), name: name.clone(), delta: String::new(), index: idx as u32,
        }));
        *entry = Some(PartialToolCall { id, name, arguments: String::new() });
    }
    if let Some(partial) = entry.as_mut() {
        if let Some(id) = dtc.id.filter(|_| partial.id.is_empty()) {
            partial.id = id;
        }
        if let Some(args) = dtc.function.and_then(|f| f.arguments).filter(|a| !a.is_empty()) {
            partial.arguments.push_str(&args);
            events.push(LlmEvent::ToolCallChunk(ToolCallChunk {
                id: partial.id.clone(), name: partial.name.clone(),
                delta: args, index: idx as u32,
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

// ── Non-streaming (complete) ──────────────────────────────────────────────────

pub(crate) async fn complete_openai_compatible(
    token:       &str,
    http:        &reqwest::Client,
    config:      &AgentConfig,
    messages:    &[Message],
    tools:       &[ToolDefinition],
    prepare_history: Option<fn(Vec<Message>) -> Vec<Message>>,
) -> Result<CompleteResponse, ApiError> {
    let tool_choice = if tools.is_empty() { None } else { Some(ToolChoice::Auto) };
    let history = match prepare_history {
        Some(f) => f(messages.to_vec()),
        None    => messages.to_vec(),
    };
    let req = request::build_oai_request(
        config, history, tools, tool_choice, false,
    );
    let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));
    let token = token.to_string();
    let body = post_json(http, &url, &req, &token, &PostConfig {
        use_query_key:  false,
        auth_header:    None,
        extra_headers:  &[],
        max_retries:    config.max_retries,
        retry_delay_ms: config.retry_delay_ms,
    }).await?;

    let raw: response::CompleteResponse = serde_json::from_str(&body)
        .map_err(ApiError::Json)?;

    let choice = raw.choices.into_iter().next();
    let finish_reason = choice.as_ref()
        .and_then(|c| c.finish_reason.as_deref())
        .map(FinishReason::from);
    let msg = choice.map(|c| c.message);

    Ok(CompleteResponse {
        content: msg.as_ref().and_then(|m| m.content.clone()),
        reasoning: None,
        tool_calls: msg.map(|m| {
            m.tool_calls.unwrap_or_default().into_iter().map(|tc| ToolCall {
                id: tc.id,
                name: tc.function.name,
                arguments: tc.function.arguments,
            }).collect()
        }).unwrap_or_default(),
        usage: raw.usage.map(UsageStats::from).unwrap_or_default(),
        finish_reason: finish_reason.unwrap_or_default(),
    })
}
