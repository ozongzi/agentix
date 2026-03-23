use std::sync::Arc;

use serde_json::Value;

use crate::request::UserContent;

/// Trait for custom message types that can be sent through the event bus.
pub trait CustomMsg: std::fmt::Debug + Send + Sync + 'static {
    fn as_any(&self) -> &dyn std::any::Any;
}

/// The unified message type for all agent and graph node communication.
///
/// The same variant carries different granularity depending on the path:
/// - **Streaming path**: `Token` is a single chunk; `subscribe_assembled()`
///   on [`EventBus`][crate::EventBus] folds many `Token`s into one.
/// - **Non-streaming / assembled path**: `Token` is the complete response.
///
/// Either way the variant is the same — no information is lost or renamed.
///
/// # Turn order
/// ```text
/// TurnStart
///   Token*        (zero or more; one in assembled view)
///   Reasoning*    (zero or more; one in assembled view)
///   ToolCall*     (zero or more, always fully assembled)
///   ToolResult*   (one per ToolCall, after execution)
/// Done
/// ```
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum Msg {
    // ── Turn envelope ────────────────────────────────────────────────────────
    /// A generation turn is starting.
    TurnStart,
    /// The generation turn (including all tool-call rounds) has completed.
    Done,

    // ── Input ────────────────────────────────────────────────────────────────
    /// A user (human-side) message submitted to the agent / node.
    /// May contain text and/or images.
    User(Vec<UserContent>),

    // ── LLM output ───────────────────────────────────────────────────────────
    /// An LLM output token.  One chunk in streaming; the full text in assembled.
    Token(String),
    /// A reasoning token (e.g. DeepSeek-R1).  Same streaming/assembled duality.
    Reasoning(String),
    /// A complete tool invocation (args fully assembled before emission).
    ToolCall { id: String, name: String, args: String },
    /// The result of a completed tool invocation.
    ToolResult { call_id: String, name: String, result: Value },

    // ── Error / extension ────────────────────────────────────────────────────
    /// An error occurred during the current turn.
    Error(String),
    /// Application-defined payload; attach typed data via [`CustomMsg`].
    Custom(Arc<dyn CustomMsg>),
}
