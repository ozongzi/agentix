use std::sync::Arc;
use serde_json::Value;
use crate::request::{UserContent, ToolCall};
use crate::types::UsageStats;

/// Trait for custom payloads carried in events.
pub trait CustomEvent: std::fmt::Debug + Send + Sync + 'static {
    fn as_any(&self) -> &dyn std::any::Any;
}

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

// ── Agent Events ─────────────────────────────────────────────────────────────

/// Events emitted by an Agent node.
/// 
/// This is what the outside world or an orchestrator sees.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Incremental response tokens.
    Token(String),
    /// Reasoning tokens.
    Reasoning(String),
    /// A fragment of a tool call being generated (useful for real-time UI).
    ToolCallChunk(crate::types::ToolCallChunk),
    /// The agent is calling a tool.
    ToolCall(ToolCall),
    /// A tool has finished, here is the result.
    ToolResult {
        call_id: String,
        name:    String,
        result:  Value,
    },
    /// Accumulated usage for the turn.
    Usage(UsageStats),
    /// The current interaction turn is complete.
    Done,
    /// An error occurred.
    Error(String),
    /// Extensibility point.
    Custom(Arc<dyn CustomEvent>),
}

// ── Agent Inputs ─────────────────────────────────────────────────────────────

/// Inputs accepted by an Agent node.
#[derive(Debug, Clone)]
pub enum AgentInput {
    /// A new message from the user.
    User(Vec<UserContent>),
    /// The result of a tool execution.
    ToolResult {
        call_id: String,
        result:  Value,
    },
    /// Hard stop: stop current LLM stream immediately.
    Abort,
}
