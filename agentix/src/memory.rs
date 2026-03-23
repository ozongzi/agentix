use async_trait::async_trait;

use crate::msg::Msg;
use crate::request::Message;

/// A component that records bus events and provides conversation context.
///
/// Implement this to customise how history is stored, compressed, or retrieved.
/// The agent loop calls [`record`] for every [`Msg`] it broadcasts and calls
/// [`context`] before each LLM request to get the message history.
///
/// # Built-in implementations
/// - [`InMemory`] — keeps all messages in a `Vec` (default)
/// - [`SlidingWindow`] — keeps the last N messages
#[async_trait]
pub trait Memory: Send {
    /// Called for every message emitted on the bus during a turn.
    async fn record(&mut self, msg: &Msg);

    /// Returns the conversation history used to build the next LLM request.
    /// Implementations may summarise, truncate, or semantically retrieve here.
    async fn context(&self) -> Vec<Message>;
}

// ── InMemory ──────────────────────────────────────────────────────────────────

/// Full in-memory history — the default [`Memory`] implementation.
pub struct InMemory {
    messages: Vec<Message>,
    /// Buffer for accumulating assistant token stream before committing.
    assistant_buf: String,
    /// Buffer for reasoning tokens.
    reasoning_buf: String,
    /// Pending tool calls (id → (name, args)).
    tool_call_bufs: std::collections::HashMap<String, (String, String)>,
}

impl Default for InMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemory {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            assistant_buf: String::new(),
            reasoning_buf: String::new(),
            tool_call_bufs: std::collections::HashMap::new(),
        }
    }

    /// Seed with existing history (e.g. loaded from a database).
    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        self.messages = history;
        self
    }
}

#[async_trait]
impl Memory for InMemory {
    async fn record(&mut self, msg: &Msg) {
        match msg {
            Msg::User(text) => {
                self.messages.push(Message::User(vec![
                    crate::request::UserContent::Text(text.clone()),
                ]));
            }
            Msg::Token(t) => {
                self.assistant_buf.push_str(t);
            }
            Msg::Reasoning(t) => {
                self.reasoning_buf.push_str(t);
            }
            Msg::ToolCall { id, name, args } => {
                let entry = self.tool_call_bufs
                    .entry(id.clone())
                    .or_insert_with(|| (name.clone(), String::new()));
                entry.1.push_str(args);
            }
            Msg::ToolResult { call_id, name, result } => {
                self.messages.push(Message::ToolResult {
                    call_id: call_id.clone(),
                    content: result.to_string(),
                });
                let _ = (name,);
            }
            Msg::Done => {
                // Commit accumulated assistant turn to history.
                let tool_calls: Vec<crate::request::ToolCall> = self
                    .tool_call_bufs
                    .drain()
                    .map(|(id, (name, arguments))| crate::request::ToolCall { id, name, arguments })
                    .collect();

                let content = if self.assistant_buf.is_empty() { None } else { Some(self.assistant_buf.clone()) };
                let reasoning = if self.reasoning_buf.is_empty() { None } else { Some(self.reasoning_buf.clone()) };

                if content.is_some() || !tool_calls.is_empty() || reasoning.is_some() {
                    self.messages.push(Message::Assistant {
                        content,
                        reasoning,
                        tool_calls,
                    });
                }

                self.assistant_buf.clear();
                self.reasoning_buf.clear();
            }
            _ => {}
        }
    }

    async fn context(&self) -> Vec<Message> {
        self.messages.clone()
    }
}

// ── SlidingWindow ─────────────────────────────────────────────────────────────

/// Keeps only the last `max` messages (by count), discarding older ones.
pub struct SlidingWindow {
    inner: InMemory,
    max:   usize,
}

impl SlidingWindow {
    pub fn new(max: usize) -> Self {
        Self { inner: InMemory::new(), max }
    }
}

#[async_trait]
impl Memory for SlidingWindow {
    async fn record(&mut self, msg: &Msg) {
        self.inner.record(msg).await;
    }

    async fn context(&self) -> Vec<Message> {
        let all = self.inner.context().await;
        let len = all.len();
        if len <= self.max {
            all
        } else {
            all[len - self.max..].to_vec()
        }
    }
}
