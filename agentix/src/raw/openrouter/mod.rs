//! OpenRouter provider — OpenAI-Chat-compatible proxy with a unified reasoning
//! abstraction across underlying models.
//!
//! # Reasoning semantics
//!
//! `LlmEvent::Reasoning(text)` streams whatever plaintext reasoning OpenRouter
//! chooses to expose — either the simple `reasoning` delta string or the
//! `text` field of a `reasoning.text` entry within `reasoning_details`. The
//! authoritative state for multi-turn tool loops is the entire
//! `reasoning_details[]` array (whose entries may be `reasoning.text` /
//! `reasoning.summary` / `reasoning.encrypted`). That round-trips via
//! [`LlmEvent::AssistantState`] → [`Message::Assistant::provider_data`]
//! (envelope tag `openrouter_reasoning_details`).

pub mod request;
pub mod response;

use std::collections::HashMap;

use eventsource_stream::Eventsource;
use futures::{StreamExt, stream::BoxStream};
use serde_json::Value;
use tracing::debug;

use crate::config::AgentConfig;
use crate::error::ApiError;
use crate::msg::LlmEvent;
use crate::provider::{PostConfig, post_json, post_streaming};
use crate::raw::shared::ToolDefinition;
use crate::request::{Message, ToolCall, ToolChoice};
use crate::types::{
    CompleteResponse, FinishReason, PartialToolCall, StreamBufs, ToolCallChunk, UsageStats,
};

use response::{DeltaToolCall, StreamChunk};

pub(crate) async fn stream_openrouter(
    token: &str,
    http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
    let tool_choice = if tools.is_empty() {
        None
    } else {
        Some(ToolChoice::Auto)
    };
    let req =
        request::build_openrouter_request(config, messages.to_vec(), tools, tool_choice, true);
    let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));
    let token = token.to_string();
    let resp = post_streaming(
        http,
        &url,
        &req,
        &token,
        &PostConfig {
            use_query_key: false,
            auth_header: None,
            extra_headers: &[],
            max_retries: config.max_retries,
            retry_delay_ms: config.retry_delay_ms,
        },
    )
    .await?;

    Ok(async_stream::stream! {
        let mut bufs = StreamBufs::new();
        let mut details = ReasoningDetailsAccumulator::default();
        let mut sse = resp.bytes_stream().eventsource();
        let mut saw_done = false;
        let mut saw_finish_reason = false;

        while let Some(ev_res) = sse.next().await {
            match ev_res {
                Ok(ev) => {
                    #[cfg(feature = "sensitive-logs")]
                    if crate::sensitive_logs_enabled() {
                        tracing::info!(body = %ev.data, "received raw streaming response chunk");
                    }
                    if ev.data == "[DONE]" {
                        saw_done = true;
                        break;
                    }
                    match serde_json::from_str::<StreamChunk>(&ev.data) {
                        Ok(chunk) => {
                            if chunk
                                .choices
                                .iter()
                                .any(|choice| choice.finish_reason.is_some())
                            {
                                saw_finish_reason = true;
                            }
                            for lev in parse_chunk(chunk, &mut bufs, &mut details) { yield lev; }
                        }
                        Err(e) => debug!(data = %ev.data, error = %e, "openrouter chunk parse failed"),
                    }
                }
                Err(e) => { yield LlmEvent::Error(e.to_string()); break; }
            }
        }
        if !saw_done && !saw_finish_reason {
            yield LlmEvent::Error("stream ended without [DONE] or finish_reason".to_string());
        }

        let tool_calls = finalize(&mut bufs);
        let finalized_details = details.into_sorted();

        for tc in &tool_calls { yield LlmEvent::ToolCall(tc.clone()); }

        // Gate: emit round-trip state only when reasoning_details was seen
        // AND there's a tool_call to continue from. Pure-reasoning turns are
        // terminal; no underlying provider needs state round-trip for those.
        if !finalized_details.is_empty() && !tool_calls.is_empty() {
            yield LlmEvent::AssistantState(serde_json::json!({
                "openrouter_reasoning_details": finalized_details,
            }));
        }

        yield LlmEvent::Done;
    }.boxed())
}

fn parse_chunk(
    chunk: StreamChunk,
    bufs: &mut StreamBufs,
    details: &mut ReasoningDetailsAccumulator,
) -> Vec<LlmEvent> {
    let mut events = Vec::new();
    if let Some(u) = chunk.usage {
        events.push(LlmEvent::Usage(UsageStats::from(u)));
    }
    let choice = match chunk.choices.into_iter().next() {
        Some(c) => c,
        None => return events,
    };
    let delta = choice.delta;
    if let Some(dtcs) = delta.tool_calls {
        for dtc in dtcs {
            events.extend(handle_tool_call_delta(dtc, bufs));
        }
    }
    if let Some(reasoning_details) = delta.reasoning_details {
        for entry in reasoning_details {
            // Also surface plaintext reasoning fragments to the user as
            // LlmEvent::Reasoning so streaming UIs show thinking even when
            // the underlying provider only emits typed details.
            if entry.get("type").and_then(|t| t.as_str()) == Some("reasoning.text")
                && let Some(text) = entry.get("text").and_then(|t| t.as_str())
                && !text.is_empty()
            {
                bufs.reasoning_buf.push_str(text);
                events.push(LlmEvent::Reasoning(text.to_string()));
            }
            details.ingest(entry);
        }
    }
    if let Some(r) = delta.reasoning.filter(|s| !s.is_empty()) {
        bufs.reasoning_buf.push_str(&r);
        events.push(LlmEvent::Reasoning(r));
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
        let id = dtc.id.clone().unwrap_or_default();
        let name = dtc
            .function
            .as_ref()
            .and_then(|f| f.name.clone())
            .unwrap_or_default();
        events.push(LlmEvent::ToolCallChunk(ToolCallChunk {
            id: id.clone(),
            name: name.clone(),
            delta: String::new(),
            index: idx as u32,
        }));
        *entry = Some(PartialToolCall {
            id,
            name,
            arguments: String::new(),
        });
    }
    if let Some(partial) = entry.as_mut() {
        if let Some(id) = dtc.id.filter(|_| partial.id.is_empty()) {
            partial.id = id;
        }
        if let Some(args) = dtc
            .function
            .and_then(|f| f.arguments)
            .filter(|a| !a.is_empty())
        {
            partial.arguments.push_str(&args);
            events.push(LlmEvent::ToolCallChunk(ToolCallChunk {
                id: partial.id.clone(),
                name: partial.name.clone(),
                delta: args,
                index: idx as u32,
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

/// Reassembles streamed `reasoning_details` fragments.
///
/// Streaming often fragments a single typed entry across chunks — the naive
/// "push each fragment" approach (LangChain #36400) produces a malformed
/// `reasoning_details[]` on round-trip. We key by the `index` field on each
/// entry and coalesce same-index fragments by string-appending their text /
/// summary / data slots; entries missing `index` fall back to append order.
#[derive(Default)]
struct ReasoningDetailsAccumulator {
    indexed: HashMap<u64, Value>,
    unindexed: Vec<Value>,
}

impl ReasoningDetailsAccumulator {
    fn ingest(&mut self, entry: Value) {
        let index = entry.get("index").and_then(|i| i.as_u64());
        match index {
            Some(idx) => match self.indexed.get_mut(&idx) {
                Some(existing) => merge_reasoning_entry(existing, &entry),
                None => {
                    self.indexed.insert(idx, entry);
                }
            },
            None => self.unindexed.push(entry),
        }
    }

    fn into_sorted(self) -> Vec<Value> {
        let mut entries: Vec<(u64, Value)> = self.indexed.into_iter().collect();
        entries.sort_by_key(|(i, _)| *i);
        let mut out: Vec<Value> = entries.into_iter().map(|(_, v)| v).collect();
        out.extend(self.unindexed);
        out
    }
}

fn merge_reasoning_entry(existing: &mut Value, delta: &Value) {
    // Append text-like payload fields, otherwise overwrite scalar identity
    // fields (id / format / type stay stable across fragments).
    for key in ["text", "summary", "data"] {
        if let Some(delta_val) = delta.get(key).and_then(|v| v.as_str()) {
            let current = existing.get(key).and_then(|v| v.as_str()).unwrap_or("");
            let merged = format!("{current}{delta_val}");
            if let Some(obj) = existing.as_object_mut() {
                obj.insert(key.to_string(), Value::String(merged));
            }
        }
    }
    // Copy over any field the existing entry doesn't yet have (e.g. late
    // arrival of `signature` or `id`).
    if let (Some(dobj), Some(eobj)) = (delta.as_object(), existing.as_object_mut()) {
        for (k, v) in dobj {
            if !eobj.contains_key(k) {
                eobj.insert(k.clone(), v.clone());
            }
        }
    }
}

// ── Non-streaming (complete) ──────────────────────────────────────────────────

pub(crate) async fn complete_openrouter(
    token: &str,
    http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<CompleteResponse, ApiError> {
    let tool_choice = if tools.is_empty() {
        None
    } else {
        Some(ToolChoice::Auto)
    };
    let req =
        request::build_openrouter_request(config, messages.to_vec(), tools, tool_choice, false);
    let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));
    let token = token.to_string();
    let body = post_json(
        http,
        &url,
        &req,
        &token,
        &PostConfig {
            use_query_key: false,
            auth_header: None,
            extra_headers: &[],
            max_retries: config.max_retries,
            retry_delay_ms: config.retry_delay_ms,
        },
    )
    .await?;

    let raw: response::CompleteResponse = serde_json::from_str(&body).map_err(ApiError::Json)?;

    let choice = raw.choices.into_iter().next();
    let finish_reason = choice
        .as_ref()
        .and_then(|c| c.finish_reason.as_deref())
        .map(FinishReason::from);
    let msg = choice.map(|c| c.message);

    let reasoning_details = msg.as_ref().and_then(|m| m.reasoning_details.clone());
    let tool_calls: Vec<ToolCall> = msg
        .as_ref()
        .and_then(|m| m.tool_calls.clone())
        .unwrap_or_default()
        .into_iter()
        .map(|tc| ToolCall {
            id: tc.id,
            name: tc.function.name,
            arguments: tc.function.arguments,
        })
        .collect();

    // Gate matches streaming path.
    let provider_data = reasoning_details
        .as_ref()
        .filter(|d| !d.is_empty() && !tool_calls.is_empty())
        .map(|d| {
            serde_json::json!({
                "openrouter_reasoning_details": d,
            })
        });

    Ok(CompleteResponse {
        content: msg.as_ref().and_then(|m| m.content.clone()),
        reasoning: msg.as_ref().and_then(|m| m.reasoning.clone()),
        tool_calls,
        provider_data,
        usage: raw.usage.map(UsageStats::from).unwrap_or_default(),
        finish_reason: finish_reason.unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_details_fragments_coalesce_by_index() {
        // Simulates the LangChain #36400 scenario: two text fragments for
        // index 0 and one encrypted entry for index 1 arriving across chunks.
        let mut acc = ReasoningDetailsAccumulator::default();
        acc.ingest(serde_json::json!({
            "type": "reasoning.text",
            "index": 0,
            "format": "anthropic-claude-v1",
            "text": "Let me"
        }));
        acc.ingest(serde_json::json!({
            "type": "reasoning.text",
            "index": 0,
            "text": " think."
        }));
        acc.ingest(serde_json::json!({
            "type": "reasoning.encrypted",
            "index": 1,
            "format": "openai-responses-v1",
            "data": "ENC_BLOB"
        }));
        acc.ingest(serde_json::json!({
            "type": "reasoning.text",
            "index": 0,
            "signature": "SIG_AFTER_TEXT"
        }));
        let out = acc.into_sorted();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["text"], "Let me think.");
        assert_eq!(out[0]["format"], "anthropic-claude-v1");
        // Signature arrived after the text; merger added it without overwrite.
        assert_eq!(out[0]["signature"], "SIG_AFTER_TEXT");
        assert_eq!(out[1]["type"], "reasoning.encrypted");
        assert_eq!(out[1]["data"], "ENC_BLOB");
    }

    #[test]
    fn unindexed_entries_fall_through() {
        let mut acc = ReasoningDetailsAccumulator::default();
        acc.ingest(serde_json::json!({"type": "reasoning.summary", "summary": "brief"}));
        let out = acc.into_sorted();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["summary"], "brief");
    }
}
