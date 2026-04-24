//! OpenAI provider — targets the Responses API (`POST /v1/responses`)
//! exclusively. Chat Completions is no longer reachable via
//! `Provider::OpenAI`; route to `Provider::OpenRouter` (with a custom
//! `base_url`) for OpenAI-compatible servers that don't implement `/responses`.
//!
//! # Reasoning semantics
//!
//! `LlmEvent::Reasoning(text)` streams OpenAI's *reasoning summary* — a
//! summarized view of the model's internal chain of thought (the full CoT
//! stays encrypted and never surfaces in plaintext). This is asymmetric with
//! Anthropic, whose `LlmEvent::Reasoning` carries the full thinking-block
//! text. For OpenAI the authoritative state used to continue a reasoning
//! chain across tool calls is the `encrypted_content` blob — captured into
//! [`Message::Assistant::provider_data`] and round-tripped verbatim via
//! [`LlmEvent::AssistantState`].

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

use request::ToolChoice;
use response::{OutputItemLite, StreamEvent};

// ── Streaming ────────────────────────────────────────────────────────────────

pub(crate) async fn stream_openai(
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
    let req = request::build_responses_request(config, messages.to_vec(), tools, tool_choice, true);
    let url = format!("{}/responses", config.base_url.trim_end_matches('/'));
    let resp = post_streaming(
        http,
        &url,
        &req,
        token,
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
        // Per-output-index bookkeeping: Responses streams several items per
        // turn (reasoning / function_call / message), keyed by `output_index`.
        let mut items: Vec<Option<StreamItemState>> = Vec::new();
        let mut sse = resp.bytes_stream().eventsource();
        let mut saw_terminal = false;
        let mut final_output: Option<Vec<serde_json::Value>> = None;

        while let Some(ev_res) = sse.next().await {
            match ev_res {
                Ok(ev) => {
                    #[cfg(feature = "sensitive-logs")]
                    if crate::sensitive_logs_enabled() {
                        tracing::info!(body = %ev.data, "received raw streaming response chunk");
                    }
                    if ev.data == "[DONE]" {
                        break;
                    }
                    match serde_json::from_str::<StreamEvent>(&ev.data) {
                        Ok(event) => {
                            let (lev, done, output_snapshot, usage) =
                                handle_stream_event(event, &mut bufs, &mut items);
                            for e in lev { yield e; }
                            if let Some(u) = usage { yield LlmEvent::Usage(u); }
                            if let Some(snapshot) = output_snapshot { final_output = Some(snapshot); }
                            if done { saw_terminal = true; break; }
                        }
                        Err(e) => debug!(data = %ev.data, error = %e, "responses chunk parse failed"),
                    }
                }
                Err(e) => { yield LlmEvent::Error(e.to_string()); break; }
            }
        }

        if !saw_terminal {
            yield LlmEvent::Error("stream ended without response.completed".to_string());
        }

        // Emit finalized tool_calls in output-index order.
        for tc in finalize_tool_calls(&mut items) {
            yield LlmEvent::ToolCall(tc);
        }

        // Round-trip gate: only emit AssistantState when the turn contains
        // BOTH reasoning AND function_call items. Pure-reasoning→message turns
        // don't need to survive — there's no follow-up tool_result referring
        // to them. Stream-error case: if response.completed never arrived,
        // `final_output` is None and the gate correctly stays closed.
        if let Some(output) = final_output
            && should_emit_state(&output)
        {
            yield LlmEvent::AssistantState(serde_json::json!({
                "openai_responses_items": output,
            }));
        }

        yield LlmEvent::Done;
    }
    .boxed())
}

// ── Non-streaming ─────────────────────────────────────────────────────────────

pub(crate) async fn complete_openai(
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
        request::build_responses_request(config, messages.to_vec(), tools, tool_choice, false);
    let url = format!("{}/responses", config.base_url.trim_end_matches('/'));
    let body = post_json(
        http,
        &url,
        &req,
        token,
        &PostConfig {
            use_query_key: false,
            auth_header: None,
            extra_headers: &[],
            max_retries: config.max_retries,
            retry_delay_ms: config.retry_delay_ms,
        },
    )
    .await?;

    // Parse twice — typed for content/tool_calls extraction, raw Value to
    // preserve the `output[]` array with `encrypted_content` on reasoning
    // items (round-tripped via provider_data on the next turn).
    let raw_value: serde_json::Value = serde_json::from_str(&body).map_err(ApiError::Json)?;
    let raw: response::Response = serde_json::from_str(&body).map_err(ApiError::Json)?;

    let mut content_buf = String::new();
    let mut reasoning_buf = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut has_reasoning = false;

    for item in &raw.output {
        match item {
            response::OutputItem::Message { content } => {
                for p in content {
                    if let response::MessageContentPart::OutputText { text } = p {
                        content_buf.push_str(text);
                    }
                }
            }
            response::OutputItem::Reasoning { summary, .. } => {
                has_reasoning = true;
                for p in summary {
                    if let response::ReasoningSummaryPart::SummaryText { text } = p {
                        reasoning_buf.push_str(text);
                    }
                }
            }
            response::OutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                tool_calls.push(ToolCall {
                    id: call_id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                });
            }
            response::OutputItem::Unknown => {}
        }
    }

    // Same gate as the streaming path: only round-trip when the turn has
    // both reasoning AND a function_call — pure reasoning→text is terminal.
    let provider_data = if has_reasoning && !tool_calls.is_empty() {
        raw_value
            .get("output")
            .and_then(|v| v.as_array())
            .map(|arr| {
                serde_json::json!({
                    "openai_responses_items": arr,
                })
            })
    } else {
        None
    };

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
        usage: raw.usage.map(UsageStats::from).unwrap_or_default(),
        finish_reason: raw
            .status
            .as_deref()
            .map(FinishReason::from)
            .unwrap_or_default(),
    })
}

// ── Stream helpers ────────────────────────────────────────────────────────────

#[derive(Debug)]
enum StreamItemState {
    Message,
    Reasoning,
    FunctionCall(PartialToolCall),
    Other,
}

fn ensure_item_slot(items: &mut Vec<Option<StreamItemState>>, idx: usize) {
    if items.len() <= idx {
        items.resize_with(idx + 1, || None);
    }
}

/// Return value: (events to yield, stream-done flag, final output snapshot if any, usage if any).
fn handle_stream_event(
    event: StreamEvent,
    bufs: &mut StreamBufs,
    items: &mut Vec<Option<StreamItemState>>,
) -> (
    Vec<LlmEvent>,
    bool,
    Option<Vec<serde_json::Value>>,
    Option<UsageStats>,
) {
    match event {
        StreamEvent::ResponseCreated
        | StreamEvent::ResponseInProgress
        | StreamEvent::OutputItemDone
        | StreamEvent::OutputTextDone
        | StreamEvent::ReasoningSummaryTextDone
        | StreamEvent::FunctionCallArgumentsDone
        | StreamEvent::Other => (vec![], false, None, None),

        StreamEvent::OutputItemAdded { output_index, item } => {
            let idx = output_index as usize;
            ensure_item_slot(items, idx);
            let state = match item {
                OutputItemLite::Message { .. } => StreamItemState::Message,
                OutputItemLite::Reasoning { .. } => StreamItemState::Reasoning,
                OutputItemLite::FunctionCall { call_id, name, .. } => {
                    StreamItemState::FunctionCall(PartialToolCall {
                        id: call_id.unwrap_or_default(),
                        name: name.unwrap_or_default(),
                        arguments: String::new(),
                    })
                }
                OutputItemLite::Unknown => StreamItemState::Other,
            };
            // Emit an empty ToolCallChunk so UIs can show the tool name early,
            // mirroring how the Chat Completions path announced tool_calls.
            let events = if let StreamItemState::FunctionCall(partial) = &state {
                vec![LlmEvent::ToolCallChunk(ToolCallChunk {
                    id: partial.id.clone(),
                    name: partial.name.clone(),
                    delta: String::new(),
                    index: output_index,
                })]
            } else {
                vec![]
            };
            items[idx] = Some(state);
            (events, false, None, None)
        }

        StreamEvent::OutputTextDelta { delta, .. } if !delta.is_empty() => {
            bufs.content_buf.push_str(&delta);
            (vec![LlmEvent::Token(delta)], false, None, None)
        }
        StreamEvent::OutputTextDelta { .. } => (vec![], false, None, None),

        StreamEvent::ReasoningSummaryTextDelta { delta, .. } if !delta.is_empty() => {
            bufs.reasoning_buf.push_str(&delta);
            (vec![LlmEvent::Reasoning(delta)], false, None, None)
        }
        StreamEvent::ReasoningSummaryTextDelta { .. } => (vec![], false, None, None),

        StreamEvent::FunctionCallArgumentsDelta {
            output_index,
            delta,
        } if !delta.is_empty() => {
            let idx = output_index as usize;
            ensure_item_slot(items, idx);
            if let Some(Some(StreamItemState::FunctionCall(partial))) = items.get_mut(idx) {
                partial.arguments.push_str(&delta);
                let chunk = ToolCallChunk {
                    id: partial.id.clone(),
                    name: partial.name.clone(),
                    delta: delta.clone(),
                    index: output_index,
                };
                return (vec![LlmEvent::ToolCallChunk(chunk)], false, None, None);
            }
            (vec![], false, None, None)
        }
        StreamEvent::FunctionCallArgumentsDelta { .. } => (vec![], false, None, None),

        StreamEvent::ResponseCompleted { response } => {
            let usage = response.usage.map(UsageStats::from);
            (vec![], true, Some(response.output), usage)
        }
        StreamEvent::ResponseFailed { response } => {
            let msg = response
                .status
                .unwrap_or_else(|| "response failed".to_string());
            (vec![LlmEvent::Error(msg)], true, None, None)
        }
        StreamEvent::Error { message } => (vec![LlmEvent::Error(message)], true, None, None),
    }
}

fn finalize_tool_calls(items: &mut [Option<StreamItemState>]) -> Vec<ToolCall> {
    items
        .iter_mut()
        .filter_map(|slot| match slot.take() {
            Some(StreamItemState::FunctionCall(p)) if !p.id.is_empty() => Some(ToolCall {
                id: p.id,
                name: p.name,
                arguments: p.arguments,
            }),
            _ => None,
        })
        .collect()
}

fn should_emit_state(output: &[serde_json::Value]) -> bool {
    let mut has_reasoning = false;
    let mut has_function_call = false;
    for item in output {
        match item.get("type").and_then(|t| t.as_str()) {
            Some("reasoning") => has_reasoning = true,
            Some("function_call") => has_function_call = true,
            _ => {}
        }
    }
    has_reasoning && has_function_call
}
