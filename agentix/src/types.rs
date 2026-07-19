//! Shared types used across the raw provider and request layers.
//!
//! These types are kept separate to avoid circular dependencies between
//! `raw/` (provider wire formats) and `request` (public API).

// ── Usage & Accounting ────────────────────────────────────────────────────────

/// Normalized token usage for a single request or an entire session,
/// expressed as **mutually exclusive billing buckets**.
///
/// Every token lands in exactly one bucket, so any aggregate is a plain sum
/// and cost is always `Σ bucket × rate` (see [`crate::pricing`]). Provider
/// adapters are responsible for converting each upstream's wire fields into
/// these buckets, subtracting where the upstream reports overlapping counts:
///
/// - **Anthropic-family** (Anthropic, MiniMax, MiMo, Claude Code): wire fields
///   are already disjoint (`input_tokens` excludes cache tokens) — copied as-is.
/// - **OpenAI-family** (OpenAI, Codex, DeepSeek, Kimi, GLM, Grok, OpenRouter):
///   cached tokens are a *subset* of `prompt_tokens` and reasoning a subset of
///   output — the adapter subtracts (`input = prompt − cached`,
///   `output = completion − reasoning`).
/// - **Gemini**: `thoughtsTokenCount` is already disjoint from
///   `candidatesTokenCount`; modality splits come from `promptTokensDetails`.
///
/// Modality buckets (`input_image` / `input_audio` / `input_video` /
/// `input_document`) are populated only by providers that report per-modality
/// counts (Gemini; OpenAI audio). Everyone else folds multimodal input into
/// `input` and leaves them at zero.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Usage {
    /// Fresh (uncached) **text** input tokens, billed at the full input rate.
    pub input: u64,
    /// Fresh image input tokens (only reported by some providers).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub input_image: u64,
    /// Fresh audio input tokens.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub input_audio: u64,
    /// Fresh video input tokens.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub input_video: u64,
    /// Fresh document (PDF) input tokens.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub input_document: u64,
    /// Number of input images, for providers that price per image
    /// (e.g. OpenRouter's `pricing.image`). Independent of token buckets.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub images: u64,
    /// Prompt-cache read tokens (billed at the cache-read rate).
    pub cache_read: u64,
    /// Prompt-cache write tokens, 5-minute TTL (Anthropic: 1.25× input).
    /// Providers that don't distinguish TTLs report all writes here.
    pub cache_write_5m: u64,
    /// Prompt-cache write tokens, 1-hour TTL (Anthropic: 2× input).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cache_write_1h: u64,
    /// Visible output tokens (text), **excluding** reasoning.
    pub output: u64,
    /// Audio output tokens (TTS-style responses).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub output_audio: u64,
    /// Reasoning/thinking output tokens, disjoint from `output`. Billed at
    /// the output rate on every surveyed provider unless the price sheet
    /// says otherwise.
    pub reasoning: u64,
    /// Cost as reported by the provider itself (e.g. OpenRouter `usage.cost`,
    /// Claude Code `total_cost_usd`). When present, this is authoritative and
    /// preferred over price-sheet estimation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reported_cost: Option<ReportedCost>,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

/// A cost figure reported by the provider on the wire, in the provider's
/// native currency (ISO-4217 code).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReportedCost {
    pub amount: f64,
    /// ISO-4217 currency code, e.g. `"USD"`.
    pub currency: String,
}

impl Usage {
    /// All fresh (uncached) input tokens across modalities.
    pub fn fresh_input(&self) -> u64 {
        self.input + self.input_image + self.input_audio + self.input_video + self.input_document
    }

    /// All cache-write tokens regardless of TTL.
    pub fn cache_write(&self) -> u64 {
        self.cache_write_5m + self.cache_write_1h
    }

    /// Total input tokens: fresh + cache read + cache write.
    pub fn total_input(&self) -> u64 {
        self.fresh_input() + self.cache_read + self.cache_write()
    }

    /// Total output tokens: visible + audio + reasoning.
    pub fn total_output(&self) -> u64 {
        self.output + self.output_audio + self.reasoning
    }

    /// Grand total across all buckets.
    pub fn total(&self) -> u64 {
        self.total_input() + self.total_output()
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        self.input += rhs.input;
        self.input_image += rhs.input_image;
        self.input_audio += rhs.input_audio;
        self.input_video += rhs.input_video;
        self.input_document += rhs.input_document;
        self.images += rhs.images;
        self.cache_read += rhs.cache_read;
        self.cache_write_5m += rhs.cache_write_5m;
        self.cache_write_1h += rhs.cache_write_1h;
        self.output += rhs.output;
        self.output_audio += rhs.output_audio;
        self.reasoning += rhs.reasoning;
        // Sum reported costs only when currencies match; on mismatch (or when
        // only one side reports) keep whichever single figure exists rather
        // than fabricating a mixed-currency sum.
        self.reported_cost = match (self.reported_cost.take(), rhs.reported_cost) {
            (Some(a), Some(b)) if a.currency == b.currency => Some(ReportedCost {
                amount: a.amount + b.amount,
                currency: a.currency,
            }),
            (Some(a), Some(_)) => Some(a),
            (Some(a), None) => Some(a),
            (None, b) => b,
        };
    }
}

// ── Finish reason ─────────────────────────────────────────────────────────────

/// Why the model stopped generating.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural end of the response.
    #[default]
    Stop,
    /// Hit the `max_tokens` limit — response may be truncated.
    Length,
    /// The model emitted one or more tool calls.
    ToolCalls,
    /// Content was filtered by the provider's safety system.
    ContentFilter,
    /// Any other provider-specific reason not covered above.
    Other(String),
}

impl FinishReason {
    /// Returns `true` if the response was truncated due to token limit.
    pub fn is_truncated(&self) -> bool {
        matches!(self, FinishReason::Length)
    }
}

impl From<&str> for FinishReason {
    fn from(s: &str) -> Self {
        match s {
            // OpenAI / Gemini
            "stop" | "STOP" => FinishReason::Stop,
            "length" | "MAX_TOKENS" => FinishReason::Length,
            "tool_calls" | "MALFORMED_FUNCTION_CALL" => FinishReason::ToolCalls,
            "content_filter" | "SAFETY" | "PROHIBITED_CONTENT" | "SPII" | "BLOCKLIST" => {
                FinishReason::ContentFilter
            }
            // Anthropic
            "end_turn" => FinishReason::Stop,
            "max_tokens" => FinishReason::Length,
            "tool_use" => FinishReason::ToolCalls,
            "stop_sequence" => FinishReason::Stop,
            other => FinishReason::Other(other.to_string()),
        }
    }
}
/// The result of a non-streaming (complete) API call.
#[derive(Debug, Clone, Default)]
pub struct CompleteResponse {
    /// The text content produced by the model (may be empty if only tool calls).
    pub content: Option<String>,
    /// Chain-of-thought / reasoning text, if any.
    pub reasoning: Option<String>,
    /// Tool calls requested by the model.
    pub tool_calls: Vec<crate::request::ToolCall>,
    /// Opaque per-turn state for providers that need round-tripping
    /// (e.g. Anthropic thinking blocks + signatures). Attach to
    /// [`crate::Message::Assistant.provider_data`] to preserve across turns.
    pub provider_data: Option<serde_json::Value>,
    /// Token usage statistics.
    pub usage: Usage,
    /// Why the model stopped generating.
    pub finish_reason: FinishReason,
}

impl CompleteResponse {
    /// Deserialize the response content as JSON into `T`.
    ///
    /// Equivalent to `serde_json::from_str(response.content.unwrap_or_default())`.
    /// Useful with [`Request::json_schema`] or [`Request::json`].
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> serde_json::Result<T> {
        serde_json::from_str(self.content.as_deref().unwrap_or(""))
    }
}

// ── Internal provider events ──────────────────────────────────────────────────

/// A tool call fragment emitted during a streaming turn.
#[derive(Debug, Clone)]
pub struct ToolCallChunk {
    /// Unique call ID assigned by the provider.
    pub id: String,
    /// Tool name being invoked.
    pub name: String,
    /// Incremental JSON argument fragment.
    pub delta: String,
    /// Zero-based index when multiple tool calls happen in one turn.
    pub index: u32,
}

// ── Streaming accumulator ─────────────────────────────────────────────────────

/// Accumulates a single tool-call's incremental SSE deltas until the stream ends.
#[derive(Debug)]
pub struct PartialToolCall {
    /// Unique call ID assigned by the provider.
    pub id: String,
    /// Tool name being invoked.
    pub name: String,
    /// JSON arguments accumulated so far.
    pub arguments: String,
}

/// Provider-agnostic streaming state — accumulates text, reasoning, and
/// tool-call fragments across SSE chunks.
pub struct StreamBufs {
    /// Accumulated text content.
    pub content_buf: String,
    /// Accumulated reasoning / chain-of-thought.
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
