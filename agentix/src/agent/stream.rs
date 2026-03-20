//! Agent streaming state machine.
//!
//! This module is responsible *only* for scheduling and polling — it does not
//! contain any business logic.  All "do actual work" functions live in
//! `executor`:
//!
//! ```text
//! AgentStream::poll_next
//!   │
//!   ├─ Idle              → spawn run_summarize future
//!   ├─ Summarizing       → poll future → ConnectingStream | FetchingResponse
//!   ├─ FetchingResponse  → poll future → YieldingToolCalls | Done  (yield Token)
//!   ├─ ConnectingStream  → poll future → StreamingChunks
//!   ├─ StreamingChunks   → poll inner stream → yield Token | YieldingToolCalls | Done
//!   ├─ YieldingToolCalls → drain queue → ExecutingTools  (yield ToolCall per item)
//!   ├─ ExecutingTools    → poll future → YieldingToolResults
//!   ├─ YieldingToolResults → drain queue → Idle  (yield ToolResult per item)
//!   └─ Done              → Poll::Ready(None)
//! ```

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::{Stream, StreamExt};

use super::executor::{
    ConnectFuture, ExecFuture, FetchFuture, StreamingData, SummarizeFuture,
    apply_chunk_delta, connect_stream, execute_tools, fetch_response, finalize_stream,
    run_summarize,
};
use crate::agent::agent_core::Agent;
use crate::types::{AgentEvent, ProviderProtocol, StreamBufs, ToolCallChunk, ToolCallResult};
use crate::error::ApiError;

// ── State machine ─────────────────────────────────────────────────────────────

/// Drives an agent through one or more API turns, tool-execution rounds, and
/// summarization passes, emitting [`AgentEvent`]s as a [`Stream`].
///
/// Obtain one by calling [`DeepSeekAgent::chat`][crate::agent::DeepSeekAgent::chat].
/// Collect it with any `futures::StreamExt` combinator or `while let Some(…)`.
///
/// # Example
///
/// ```no_run
/// use futures::StreamExt;
/// use agentix::{DeepSeekAgent, AgentEvent};
///
/// # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut stream = DeepSeekAgent::new("sk-...")
///     .streaming()
///     .chat("What is 2 + 2?");
///
/// while let Some(event) = stream.next().await {
///     match event? {
///         AgentEvent::Token(text) => print!("{text}"),
///         AgentEvent::ToolCall(c) => print!("{}", c.delta),
///         AgentEvent::ToolResult(res) => println!("[result: {}]", res.result),
///         AgentEvent::ReasoningToken(text) => print!("{text}"),
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub struct AgentStream<P: ProviderProtocol> {
    /// The agent is held here whenever no future has taken ownership of it.
    agent: Option<Agent<P>>,
    state: AgentStreamState<P>,
    /// Small queue for cases where one logical response produces multiple events
    /// (e.g. non-streaming deepseek-reasoner: ReasoningToken then Token).
    pending_events: VecDeque<AgentEvent>,
}

/// Every variant is self-contained: it either holds the agent directly or stores
/// a future that will return the agent when it resolves.
pub(crate) enum AgentStreamState<P: ProviderProtocol> {
    /// Waiting to start (or restart after tool results are delivered).
    Idle,
    /// Running `maybe_summarize` before the next API turn.
    Summarizing(SummarizeFuture<P>),
    /// Awaiting a non-streaming API response.
    FetchingResponse(FetchFuture<P>),
    /// Awaiting the initial SSE connection.
    ConnectingStream(ConnectFuture<P>),
    /// Polling an active SSE stream chunk-by-chunk.
    StreamingChunks(Box<StreamingData<P>>),
    /// Yielding individual `ToolCall` events before execution starts.
    /// `from_streaming`: if true, events were already emitted as chunks during
    /// [`StreamingChunks`] — skip emitting and go straight to [`ExecutingTools`].
    YieldingToolCalls {
        pending: VecDeque<crate::request::ToolCall>,
        raw: Vec<crate::request::ToolCall>,
        from_streaming: bool,
    },
    /// Awaiting parallel/sequential tool execution.
    ExecutingTools(ExecFuture<P>),
    /// Yielding individual `ToolResult` events after execution completes.
    YieldingToolResults { pending: VecDeque<ToolCallResult> },
    /// Terminal state — the stream will never produce another item.
    Done,
}

// ── Constructor / accessor ────────────────────────────────────────────────────

impl<P: ProviderProtocol> AgentStream<P> {
    /// Wrap an agent and start in the `Idle` state.
    pub fn new(agent: Agent<P>) -> Self {
        Self {
            agent: Some(agent),
            state: AgentStreamState::Idle,
            pending_events: VecDeque::new(),
        }
    }

    /// Consume the stream and return the agent.
    ///
    /// If the stream finished normally (or was dropped mid-stream), the agent is
    /// returned so callers can continue the conversation without constructing a
    /// new one.
    ///
    /// Returns `None` only if the agent is currently owned by an in-progress
    /// future (i.e. the stream was dropped mid-poll, which is very unusual).
    pub fn into_agent(self) -> Option<Agent<P>> {
        match self.state {
            AgentStreamState::StreamingChunks(data) => Some(data.agent),
            _ => self.agent,
        }
    }
}

// ── Stream implementation ─────────────────────────────────────────────────────

impl<P: ProviderProtocol> Stream for AgentStream<P> {
    type Item = Result<AgentEvent, ApiError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            // Drain any events queued by a previous iteration before polling state.
            if let Some(ev) = this.pending_events.pop_front() {
                return Poll::Ready(Some(Ok(ev)));
            }

            // ── StreamingChunks is handled first to avoid borrow-checker
            //    conflicts: we need to both poll the inner stream *and* replace
            //    `this.state`, which requires owning the data.
            if matches!(this.state, AgentStreamState::StreamingChunks(_)) {
                let mut data = match std::mem::replace(&mut this.state, AgentStreamState::Done) {
                    AgentStreamState::StreamingChunks(d) => d,
                    _ => unreachable!(),
                };

                match data.stream.poll_next_unpin(cx) {
                    Poll::Pending => {
                        this.state = AgentStreamState::StreamingChunks(data);
                        return Poll::Pending;
                    }

                    Poll::Ready(Some(Ok(chunk))) => {
                        let mut events = apply_chunk_delta(&mut data, chunk);
                        this.state = AgentStreamState::StreamingChunks(data);
                        // Queue all events; drain them one per poll via pending_events.
                        if !events.is_empty() {
                            // Push tail events into the pending queue (they will be
                            // returned on subsequent poll_next calls before we poll
                            // the underlying stream again).
                            for extra in events.drain(1..) {
                                this.pending_events.push_back(extra);
                            }
                            let ev = events.swap_remove(0);
                            return Poll::Ready(Some(Ok(ev)));
                        }
                        continue;
                    }

                    Poll::Ready(Some(Err(e))) => {
                        // Stream errored — salvage the agent and terminate.
                        this.agent = Some(data.agent);
                        // state stays Done (set above via mem::replace)
                        return Poll::Ready(Some(Err(e)));
                    }

                    Poll::Ready(None) => {
                        // SSE stream ended — assemble full tool calls from buffers.
                        let tool_calls = finalize_stream(&mut data);

                        if tool_calls.is_empty() {
                            this.agent = Some(data.agent);
                            this.state = AgentStreamState::Done;
                            return Poll::Ready(None);
                        }

                        this.agent = Some(data.agent);
                        this.state = AgentStreamState::YieldingToolCalls {
                            pending: VecDeque::new(),
                            raw: tool_calls,
                            from_streaming: true,
                        };
                        continue;
                    }
                }
            }

            // ── All other states ──────────────────────────────────────────────
            match &mut this.state {
                AgentStreamState::Done => return Poll::Ready(None),

                AgentStreamState::Idle => {
                    let agent = this.agent.as_mut().expect("agent missing in Idle state");
                    agent.drain_interrupts();
                    agent.drain_tool_injections();
                    let agent = this.agent.take().unwrap();
                    this.state = AgentStreamState::Summarizing(Box::pin(run_summarize(agent)));
                }

                AgentStreamState::Summarizing(fut) => match fut.as_mut().poll(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(agent) => {
                        this.state = if agent.streaming {
                            AgentStreamState::ConnectingStream(Box::pin(connect_stream(agent)))
                        } else {
                            AgentStreamState::FetchingResponse(Box::pin(fetch_response(agent)))
                        };
                    }
                },

                AgentStreamState::FetchingResponse(fut) => match fut.as_mut().poll(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready((Err(e), agent)) => {
                        this.agent = Some(agent);
                        this.state = AgentStreamState::Done;
                        return Poll::Ready(Some(Err(e)));
                    }
                    Poll::Ready((Ok(fetch), agent)) => {
                        this.agent = Some(agent);

                        if fetch.tool_calls.is_empty() {
                            this.state = AgentStreamState::Done;
                            // Queue all events yielded by the response.
                            for ev in fetch.events {
                                this.pending_events.push_back(ev);
                            }
                            // The pending_events drain at the top of the loop will emit them.
                            continue;
                        }

                        // Yield any non-tool events before transitioning.
                        for ev in fetch.events {
                            this.pending_events.push_back(ev);
                        }
                        
                        let pending = fetch
                            .tool_calls
                            .iter()
                            .cloned()
                            .collect::<VecDeque<_>>();
                        this.state = AgentStreamState::YieldingToolCalls {
                            pending,
                            raw: fetch.tool_calls,
                            from_streaming: false,
                        };

                        // The loop will continue and pick up events from pending_events.
                        continue;
                    }
                },

                AgentStreamState::ConnectingStream(fut) => match fut.as_mut().poll(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready((Err(e), agent)) => {
                        this.agent = Some(agent);
                        this.state = AgentStreamState::Done;
                        return Poll::Ready(Some(Err(e)));
                    }
                    Poll::Ready((Ok(stream), agent)) => {
                        this.state = AgentStreamState::StreamingChunks(Box::new(StreamingData {
                            stream,
                            agent,
                            bufs: StreamBufs::default(),
                        }));
                        // Loop back to hit the StreamingChunks branch.
                    }
                },

                AgentStreamState::YieldingToolCalls {
                    pending,
                    raw,
                    from_streaming,
                } => {
                    if !*from_streaming && let Some(tc) = pending.pop_front() {
                        return Poll::Ready(Some(Ok(AgentEvent::ToolCall(ToolCallChunk {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            delta: tc.arguments.clone(),
                            index: 0,
                        }))));
                    }
                    // All events yielded (or streaming — already emitted as chunks).
                    let agent = this
                        .agent
                        .take()
                        .expect("agent missing in YieldingToolCalls");
                    let raw_calls = std::mem::take(raw);
                    this.state =
                        AgentStreamState::ExecutingTools(Box::pin(execute_tools(agent, raw_calls)));
                }

                AgentStreamState::ExecutingTools(fut) => match fut.as_mut().poll(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready((tools_result, agent)) => {
                        this.agent = Some(agent);
                        this.state = AgentStreamState::YieldingToolResults {
                            pending: tools_result.results.into_iter().collect(),
                        };
                    }
                },

                AgentStreamState::YieldingToolResults { pending } => {
                    if let Some(result) = pending.pop_front() {
                        return Poll::Ready(Some(Ok(AgentEvent::ToolResult(result))));
                    }
                    // All results delivered — loop back for the next API turn.
                    this.state = AgentStreamState::Idle;
                }

                // Handled in the dedicated block above; this arm is unreachable
                // but the compiler cannot verify that without exhaustiveness help.
                AgentStreamState::StreamingChunks(_) => unreachable!(),
            }
        }
    }
}

