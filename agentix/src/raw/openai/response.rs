//! Response wire format for OpenAI's Responses API.
//!
//! Everything lives in `output[]` as one of a handful of item types. Usage
//! keys are `input_tokens` / `output_tokens` (not `prompt_tokens` /
//! `completion_tokens` like Chat Completions).

use serde::Deserialize;
use serde_json::Value;

// ── Non-streaming response ────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct Response {
    /// Stop reason echoed from the API (`"completed"`, `"max_output_tokens"`,
    /// `"failed"`, etc.). We translate into [`crate::types::FinishReason`].
    #[serde(default)]
    pub status: Option<String>,
    pub output: Vec<OutputItem>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    /// Assistant text output.
    Message {
        #[serde(default)]
        content: Vec<MessageContentPart>,
    },
    /// Encrypted reasoning trace (ID + optional plaintext summary).
    Reasoning {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        summary: Vec<ReasoningSummaryPart>,
    },
    /// Function/tool call.
    FunctionCall {
        #[serde(default)]
        id: Option<String>,
        call_id: String,
        name: String,
        arguments: String,
    },
    /// Unknown item type — tolerated via `#[serde(other)]`. Responses API adds
    /// new item kinds over time (web_search_call, computer_use_call, etc.);
    /// we ignore them for now and let the raw-JSON provider_data round-trip
    /// handle unknown shapes.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContentPart {
    OutputText {
        text: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningSummaryPart {
    SummaryText {
        text: String,
    },
    #[serde(other)]
    Unknown,
}

// ── Usage ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
    #[serde(default)]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(default)]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Debug, Deserialize, Default)]
pub struct InputTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
    /// Cache-write tokens, surfaced on the newest model families that charge
    /// cache writes (1.25× input). Zero/absent elsewhere.
    #[serde(default)]
    pub cache_write_tokens: u32,
    #[serde(default)]
    pub audio_tokens: u32,
}

#[derive(Debug, Deserialize, Default)]
pub struct OutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: u32,
    #[serde(default)]
    pub audio_tokens: u32,
}

// OpenAI billing semantics: `cached_tokens` is a SUBSET of `input_tokens`
// (billed input = (input − cached) × full + cached × cached rate) and
// `reasoning_tokens` a subset of `output_tokens`, so both are subtracted out
// to yield disjoint buckets. Audio tokens are likewise subsets with their own
// rates.
impl From<Usage> for crate::types::Usage {
    fn from(u: Usage) -> Self {
        let ind = u.input_tokens_details.unwrap_or_default();
        let outd = u.output_tokens_details.unwrap_or_default();
        // cache_write_tokens are fresh input tokens billed at the write rate
        // (1.25×) instead of the plain input rate — also a subset of
        // `input_tokens`, so they move out of the text bucket too.
        let input = (u.input_tokens as u64)
            .saturating_sub(ind.cached_tokens as u64)
            .saturating_sub(ind.audio_tokens as u64)
            .saturating_sub(ind.cache_write_tokens as u64);
        let output = (u.output_tokens as u64)
            .saturating_sub(outd.reasoning_tokens as u64)
            .saturating_sub(outd.audio_tokens as u64);
        Self {
            input,
            input_audio: ind.audio_tokens as u64,
            cache_read: ind.cached_tokens as u64,
            cache_write_5m: ind.cache_write_tokens as u64,
            output,
            output_audio: outd.audio_tokens as u64,
            reasoning: outd.reasoning_tokens as u64,
            ..Default::default()
        }
    }
}

// ── Streaming events ──────────────────────────────────────────────────────────
//
// The Responses SSE stream is verbose — dozens of event types. We only
// decode the subset we act on; everything else lands on the `Other` variant
// and is ignored.

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "response.created")]
    ResponseCreated,
    #[serde(rename = "response.in_progress")]
    ResponseInProgress,

    /// A new output item is opening. `output_index` identifies its position
    /// in the final `output[]` array — we key per-item state by it.
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        output_index: u32,
        item: OutputItemLite,
    },
    /// An output item is finished — we don't need the payload here; the
    /// added/delta events carry everything we stream, and the completed
    /// payload re-emits the full output array.
    #[serde(rename = "response.output_item.done")]
    OutputItemDone,

    /// Incremental assistant text.
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta { output_index: u32, delta: String },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone,

    /// Incremental reasoning summary text. The summary is a summarized view
    /// of the model's chain of thought (full CoT is never exposed in plain
    /// text — only as `encrypted_content` on the final completed payload).
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta { output_index: u32, delta: String },
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone,

    /// Incremental function-call arguments (accumulates into a JSON string).
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta { output_index: u32, delta: String },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone,

    /// Terminal event — carries the full `response.output[]` array, with
    /// `encrypted_content` on reasoning items. This is the only place the
    /// encrypted blob appears in the stream.
    #[serde(rename = "response.completed")]
    ResponseCompleted { response: CompletedResponse },

    #[serde(rename = "response.failed")]
    ResponseFailed { response: CompletedResponse },

    #[serde(rename = "error")]
    Error { message: String },

    /// Fallback for event types we don't consume (there are many).
    #[serde(other)]
    Other,
}

/// Lightweight item stub sent with `response.output_item.added`. We only need
/// the discriminator + identifiers to key per-item streaming state.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItemLite {
    Message {
        #[serde(default)]
        id: Option<String>,
    },
    Reasoning {
        #[serde(default)]
        id: Option<String>,
    },
    FunctionCall {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        call_id: Option<String>,
        #[serde(default)]
        name: Option<String>,
    },
    #[serde(other)]
    Unknown,
}

/// The `response` echo in `response.completed` / `response.failed`. We parse
/// just enough structurally to pull usage + status; the raw output array is
/// captured separately as `serde_json::Value` for provider_data round-trip.
#[derive(Debug, Deserialize)]
pub struct CompletedResponse {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub output: Vec<Value>,
    #[serde(default)]
    pub usage: Option<Usage>,
}
