//! Shared types used across the agent, raw provider, and request layers.
//!
//! These types are kept separate to avoid circular dependencies between
//! `raw/` (provider wire formats) and `agent/` (agent logic).

use serde_json::Value;

// ── Agent event types ─────────────────────────────────────────────────────────

/// A tool call fragment emitted during a streaming turn.
///
/// In streaming mode multiple `ToolCallChunk`s are emitted per tool call:
/// the first has an empty `delta` (name is known, no args yet); subsequent
/// chunks carry incremental argument JSON. In non-streaming mode a single
/// chunk is emitted with the complete argument JSON in `delta`.
#[derive(Debug, Clone)]
pub struct ToolCallChunk {
    pub id: String,
    pub name: String,
    pub delta: String,
    pub index: u32,
}

/// The result of a completed tool invocation.
#[derive(Debug, Clone)]
pub struct ToolCallResult {
    pub id: String,
    pub name: String,
    pub args: String,
    pub result: Value,
}

/// Events emitted by [`AgentStream`][crate::agent::AgentStream].
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A text fragment from the assistant.
    Token(String),
    /// Reasoning/thinking content (e.g. deepseek-reasoner, claude extended thinking).
    ReasoningToken(String),
    /// A tool call fragment. Accumulate `delta` values by `id` to reconstruct args.
    ToolCall(ToolCallChunk),
    /// A tool has finished executing.
    ToolResult(ToolCallResult),
}

// ── Streaming accumulator ─────────────────────────────────────────────────────

/// Accumulates a single tool-call's incremental SSE deltas until the stream ends.
#[derive(Debug)]
pub struct PartialToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Provider-agnostic streaming state, held inside `StreamingData<P>`.
/// Separated so `ProviderProtocol::parse_chunk` can mutate it without
/// knowing about `Agent<P>`.
pub struct StreamBufs {
    pub content_buf: String,
    pub reasoning_buf: String,
    /// Sparse per-index partial tool-call buffers.
    pub tool_call_bufs: Vec<Option<PartialToolCall>>,
}

impl StreamBufs {
    pub fn new() -> Self {
        Self {
            content_buf: String::new(),
            reasoning_buf: String::new(),
            tool_call_bufs: Vec::new(),
        }
    }
}

impl Default for StreamBufs {
    fn default() -> Self {
        Self::new()
    }
}

// ── ProviderProtocol ─────────────────────────────────────────────────────────

/// Implemented by provider marker types (`OpenAI`, `Anthropic`, `Gemini`).
/// Connects the generic agent machinery to provider-specific wire formats.
pub trait ProviderProtocol: Send + Sync + Unpin + 'static {
    /// The serialisable request body sent to the provider.
    type RawRequest: serde::Serialize + Send + Sync;
    /// The deserialised non-streaming response body.
    type RawResponse: for<'de> serde::Deserialize<'de> + Send + Sync;
    /// A single deserialised streaming chunk / event.
    type RawChunk: for<'de> serde::Deserialize<'de> + Send + Sync;

    /// Convert a provider-agnostic `AgentRequest` into the provider's wire format.
    fn build_raw(req: crate::request::Request) -> Self::RawRequest;

    /// Parse a complete response into agent events and raw tool calls.
    ///
    /// Returns `(events, tool_calls)` where `tool_calls` uses the unified
    /// `request::ToolCall` type so the executor can dispatch them uniformly.
    fn parse_response(raw: Self::RawResponse) -> (Vec<AgentEvent>, Vec<crate::request::ToolCall>);

    /// Apply a single streaming chunk to the accumulator buffers and return
    /// any events to yield immediately.
    fn parse_chunk(chunk: Self::RawChunk, bufs: &mut StreamBufs) -> Vec<AgentEvent>;

    /// Assemble complete tool calls from the accumulated buffers and return them.
    fn finalize_stream(bufs: &mut StreamBufs) -> Vec<crate::request::ToolCall>;

    /// Return the URL suffix for this provider's completions endpoint.
    /// Defaults to "/chat/completions" (OpenAI standard).
    fn url_suffix(model: &str, streaming: bool) -> String {
        let _ = (model, streaming);
        "/chat/completions".to_string()
    }

    /// Extra HTTP headers sent with every request (e.g. `anthropic-version`).
    /// Returns a static slice of (name, value) pairs.
    fn extra_headers() -> &'static [(&'static str, &'static str)] {
        &[]
    }

    /// If `Some(name)`, the token is sent as the given header instead of
    /// `Authorization: Bearer`. Used by Anthropic (`x-api-key`).
    fn auth_header_name() -> Option<&'static str> {
        None
    }

    /// If `true`, the token is appended as `?key=TOKEN` query parameter
    /// instead of an `Authorization` header. Used by Gemini.
    fn uses_query_key_auth() -> bool {
        false
    }

    /// Provider-specific history preparation applied before building each request.
    /// Default is a no-op. DeepSeek overrides this to enforce reasoning_content rules.
    fn prepare_history(messages: Vec<crate::request::Message>) -> Vec<crate::request::Message> {
        messages
    }

    /// Default base URL for this provider. Used by `ApiClient::new()` so that
    /// provider-specific agents get the right endpoint without requiring an
    /// explicit `.with_base_url()` call.
    ///
    /// Every built-in provider overrides this. Custom providers **must** override
    /// it too — there is no sensible default that applies to all providers.
    fn default_base_url() -> &'static str;
}
