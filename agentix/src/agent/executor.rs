//! Agent executor — pure business logic for driving one API turn.
//!
//! | Function | Responsibility |
//! |---|---|
//! | [`build_request`] | Assemble an [`Request`] from current history + tools. |
//! | [`run_summarize`] | Run the summarizer and hand the agent back. |
//! | [`fetch_response`] | Non-streaming API call; returns content + raw tool calls. |
//! | [`connect_stream`] | Open an SSE stream and hand back the `BoxStream`. |
//! | [`execute_tools`] | Dispatch all pending tool calls and collect results. |
//! | [`finalize_stream`] | Assemble tool calls from SSE buffers; record assistant turn. |
//! | [`apply_chunk_delta`] | Feed one SSE chunk into the accumulator. |

use futures::stream::BoxStream;
use serde_json::Value;

use crate::agent::agent_core::Agent;
use crate::error::ApiError;
use crate::request::{
    Request, ToolChoice, Message, ToolCall as AgentToolCall, UserContent,
};
use crate::tool_trait::Tool;
use crate::types::{AgentEvent, ProviderProtocol, StreamBufs, ToolCallResult};

// ── Internal result types ─────────────────────────────────────────────────────

pub(crate) struct FetchResult {
    pub(crate) events: Vec<AgentEvent>,
    pub(crate) tool_calls: Vec<AgentToolCall>,
}

pub(crate) struct ToolsResult {
    pub(crate) results: Vec<ToolCallResult>,
}

// ── Streaming accumulator ─────────────────────────────────────────────────────

pub(crate) struct StreamingData<P: ProviderProtocol> {
    pub(crate) stream: BoxStream<'static, Result<P::RawChunk, ApiError>>,
    pub(crate) agent: Agent<P>,
    pub(crate) bufs: StreamBufs,
}

// ── Future type aliases ───────────────────────────────────────────────────────

pub(crate) type FetchFuture<P> = std::pin::Pin<
    Box<dyn std::future::Future<Output = (Result<FetchResult, ApiError>, Agent<P>)> + Send>,
>;

pub(crate) type ConnectFuture<P> = std::pin::Pin<
    Box<
        dyn std::future::Future<
                Output = (
                    Result<
                        BoxStream<'static, Result<<P as ProviderProtocol>::RawChunk, ApiError>>,
                        ApiError,
                    >,
                    Agent<P>,
                ),
            > + Send,
    >,
>;

pub(crate) type ExecFuture<P> =
    std::pin::Pin<Box<dyn std::future::Future<Output = (ToolsResult, Agent<P>)> + Send>>;

pub(crate) type SummarizeFuture<P> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Agent<P>> + Send>>;

// ── Business-logic functions ──────────────────────────────────────────────────

/// Assemble an [`Request`] from the agent's current history and tools.
///
/// `P::prepare_history` is called on a clone of the history before building
/// the request so providers can apply wire-level rules (e.g. DeepSeek's
/// reasoning_content constraints) without mutating the stored history.
pub(crate) fn build_request<P: ProviderProtocol>(agent: &Agent<P>) -> Request {
    let messages = P::prepare_history(agent.history.clone());

    let raw_tools: Vec<_> = agent.tool_bundle.raw_tools();
    let tools = if raw_tools.is_empty() {
        None
    } else {
        Some(raw_tools)
    };
    let tool_choice = if agent.tool_bundle.is_empty() {
        None
    } else {
        Some(ToolChoice::Auto)
    };

    Request {
        system_message: agent.system_prompt.clone(),
        messages,
        model: agent.model.clone(),
        tools,
        tool_choice,
        stream: false,
        temperature: agent.temperature,
        max_tokens: agent.max_tokens,
        response_format: agent.response_format.clone(),
        extra_body: agent.extra_body.clone(),
    }
}

/// Run the agent's summarizer if enabled, then return the agent.
pub(crate) async fn run_summarize<P: ProviderProtocol>(mut agent: Agent<P>) -> Agent<P> {
    if agent.auto_summary && agent.summarizer.should_summarize(&agent.history) {
        let _ = agent.summarizer.summarize(&mut agent.history).await;
    }
    agent
}

/// Perform a single non-streaming API turn.
///
/// Returns `(Result<FetchResult, ApiError>, Agent<P>)` — ownership is always
/// transferred back so the state machine can store the agent regardless of outcome.
pub(crate) async fn fetch_response<P: ProviderProtocol>(
    mut agent: Agent<P>,
) -> (Result<FetchResult, ApiError>, Agent<P>) {
    let req = build_request(&agent);

    let resp = match agent.client.send(req).await {
        Ok(r) => r,
        Err(e) => return (Err(e), agent),
    };

    let (events, tool_calls) = P::parse_response(resp);

    let mut content: Option<String> = None;
    let mut reasoning: Option<String> = None;
    for ev in &events {
        match ev {
            AgentEvent::Token(t) => content.get_or_insert_with(String::new).push_str(t),
            AgentEvent::ReasoningToken(t) => reasoning.get_or_insert_with(String::new).push_str(t),
            _ => {}
        }
    }

    agent.history.push(Message::Assistant {
        content,
        reasoning,
        tool_calls: tool_calls.clone(),
    });

    (Ok(FetchResult { events, tool_calls }), agent)
}

/// Open an SSE stream for the current turn and return it alongside the agent.
pub(crate) async fn connect_stream<P: ProviderProtocol>(
    agent: Agent<P>,
) -> (
    Result<BoxStream<'static, Result<P::RawChunk, ApiError>>, ApiError>,
    Agent<P>,
) {
    let mut req = build_request(&agent);
    req.stream = true;
    match agent.client.clone().into_stream(req).await {
        Ok(stream) => (Ok(stream), agent),
        Err(e) => (Err(e), agent),
    }
}

/// Execute all pending tool calls in parallel and collect results.
///
/// All calls are dispatched concurrently via `join_all`; results are then
/// collected in the original order and pushed to history together, so the
/// history remains consistent regardless of completion order.
/// Interrupts are drained once after all calls finish.
pub(crate) async fn execute_tools<P: Send + 'static>(
    mut agent: Agent<P>,
    tool_calls: Vec<AgentToolCall>,
) -> (ToolsResult, Agent<P>) {
    use std::sync::Arc;
    use futures::future::join_all;

    let bundle = Arc::new(&agent.tool_bundle);

    let futures: Vec<_> = tool_calls
        .iter()
        .map(|tc| {
            let bundle = Arc::clone(&bundle);
            let name = tc.name.clone();
            let args: Value = serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);
            async move { bundle.call(&name, args).await }
        })
        .collect();

    let call_results: Vec<Value> = join_all(futures).await;

    let mut results = Vec::with_capacity(tool_calls.len());
    for (tc, result) in tool_calls.into_iter().zip(call_results) {
        agent.history.push(Message::ToolResult {
            call_id: tc.id.clone(),
            content: result.to_string(),
        });
        results.push(ToolCallResult {
            id: tc.id,
            name: tc.name,
            args: tc.arguments,
            result,
        });
    }

    while let Ok(msg) = agent.interrupt_rx.try_recv() {
        agent.history.push(Message::User(vec![UserContent::Text(msg)]));
    }

    (ToolsResult { results }, agent)
}

/// Finalize a completed SSE stream: assemble tool calls from buffers and record
/// the assistant turn in history.
pub(crate) fn finalize_stream<P: ProviderProtocol>(
    data: &mut StreamingData<P>,
) -> Vec<AgentToolCall> {
    let tool_calls = P::finalize_stream(&mut data.bufs);

    data.agent.history.push(Message::Assistant {
        content: if data.bufs.content_buf.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut data.bufs.content_buf))
        },
        reasoning: if data.bufs.reasoning_buf.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut data.bufs.reasoning_buf))
        },
        tool_calls: tool_calls.clone(),
    });

    tool_calls
}

/// Feed one SSE chunk into the accumulator and return any events to yield.
pub(crate) fn apply_chunk_delta<P: ProviderProtocol>(
    data: &mut StreamingData<P>,
    chunk: P::RawChunk,
) -> Vec<AgentEvent> {
    P::parse_chunk(chunk, &mut data.bufs)
}
