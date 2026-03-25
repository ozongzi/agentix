use crate::request::ToolCall;
use crate::types::UsageStats;

// ── LLM Provider Events ──────────────────────────────────────────────────────

/// Raw events emitted by an LLM Provider.
#[derive(Debug, Clone)]
pub enum LlmEvent {
    /// A text fragment.
    Token(String),
    /// A reasoning/thinking fragment.
    Reasoning(String),
    /// A tool call fragment emitted during streaming.
    ToolCallChunk(crate::types::ToolCallChunk),
    /// A tool call requested by the model.
    ToolCall(ToolCall),
    /// Usage statistics (usually sent at the end).
    Usage(UsageStats),
    /// The stream has ended.
    Done,
    /// A provider-level error.
    Error(String),
}
