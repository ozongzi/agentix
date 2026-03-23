use async_trait::async_trait;
use futures::StreamExt;
use tracing::{debug, warn};

use crate::msg::Msg;
use crate::request::{Message, Request};
use crate::client::LlmClient;

/// A component that records bus events and provides conversation context.
///
/// Implement this to customise how history is stored, compressed, or retrieved.
/// The agent loop calls [`record`] for every [`Msg`] it broadcasts and calls
/// [`context`] before each LLM request to get the message history.
///
/// # Built-in implementations
/// - [`InMemory`] — keeps all messages in a `Vec` (default)
/// - [`SlidingWindow`] — keeps the last N messages
/// - [`TokenSlidingWindow`] — keeps messages up to a Token limit
/// - [`LlmSummarizer`] — summarises old messages using an LLM
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

    /// Access raw messages.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Set messages directly (used by summarizers).
    pub fn set_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }
}

#[async_trait]
impl Memory for InMemory {
    async fn record(&mut self, msg: &Msg) {
        match msg {
            Msg::User(parts) => {
                self.messages.push(Message::User(parts.clone()));
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
                let mut tool_calls: Vec<crate::request::ToolCall> = self
                    .tool_call_bufs
                    .drain()
                    .map(|(id, (name, arguments))| crate::request::ToolCall { id, name, arguments })
                    .collect();
                
                // Sort by ID to ensure deterministic order if multiple tools were called.
                tool_calls.sort_by(|a, b| a.id.cmp(&b.id));

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

// ── TokenSlidingWindow ────────────────────────────────────────────────────────

/// Keeps messages up to a maximum Token limit, discarding oldest messages first.
pub struct TokenSlidingWindow {
    inner: InMemory,
    max_tokens: usize,
}

impl TokenSlidingWindow {
    pub fn new(max_tokens: usize) -> Self {
        Self { inner: InMemory::new(), max_tokens }
    }
}

#[async_trait]
impl Memory for TokenSlidingWindow {
    async fn record(&mut self, msg: &Msg) {
        self.inner.record(msg).await;
    }

    async fn context(&self) -> Vec<Message> {
        let all = self.inner.context().await;
        let mut total = 0;
        let mut result = Vec::new();

        // Iterate backwards to keep the most recent messages
        for msg in all.into_iter().rev() {
            let tokens = msg.estimate_tokens();
            if total + tokens > self.max_tokens && !result.is_empty() {
                break;
            }
            total += tokens;
            result.push(msg);
        }
        result.reverse();
        result
    }
}

// ── LlmSummarizer ─────────────────────────────────────────────────────────────

/// Summarises older conversation history using an LLM when a Token limit is reached.
pub struct LlmSummarizer {
    inner: InMemory,
    client: LlmClient,
    /// Threshold to trigger summarization.
    trigger_at_tokens: usize,
    /// Keep at least this many recent messages untouched.
    keep_recent: usize,
}

impl LlmSummarizer {
    pub fn new(client: LlmClient, trigger_at_tokens: usize) -> Self {
        Self {
            inner: InMemory::new(),
            client,
            trigger_at_tokens,
            keep_recent: 4,
        }
    }

    pub fn with_keep_recent(mut self, n: usize) -> Self {
        self.keep_recent = n;
        self
    }

    async fn summarize_if_needed(&mut self) {
        let messages = self.inner.messages();
        if messages.len() <= self.keep_recent {
            return;
        }

        let total_tokens: usize = messages.iter().map(|m| m.estimate_tokens()).sum();
        if total_tokens < self.trigger_at_tokens {
            return;
        }

        debug!(total_tokens, threshold = self.trigger_at_tokens, "triggering LLM summarization");

        let to_summarize = &messages[..messages.len() - self.keep_recent];
        let recent = &messages[messages.len() - self.keep_recent..];

        let mut summary_req = Request::default();
        summary_req.system_message = Some("You are a helpful assistant. Summarize the following conversation history concisely while preserving key facts and state.".to_string());
        summary_req.messages = to_summarize.to_vec();
        summary_req.max_tokens = Some(512);

        // We use a simplified non-streaming call for summarization
        // Since Agent handles the complexity, we'll try to get a simple response.
        let mut stream = match self.client.stream(&summary_req.messages, &[]).await {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "summarization failed to start");
                return;
            }
        };

        let mut summary_text = String::new();
        while let Some(msg) = stream.next().await {
            match msg {
                Msg::Token(t) => summary_text.push_str(&t),
                Msg::Error(e) => {
                    warn!(error = %e, "summarization stream error");
                    return;
                }
                Msg::Done => break,
                _ => {}
            }
        }

        if summary_text.is_empty() {
            warn!("summarization returned empty text");
            return;
        }

        let mut new_history = Vec::new();
        new_history.push(Message::Assistant {
            content: Some(format!("[auto-summary] {}", summary_text)),
            reasoning: None,
            tool_calls: vec![],
        });
        new_history.extend_from_slice(recent);

        self.inner.set_messages(new_history);
    }
}

#[async_trait]
impl Memory for LlmSummarizer {
    async fn record(&mut self, msg: &Msg) {
        self.inner.record(msg).await;
        if matches!(msg, Msg::Done) {
            self.summarize_if_needed().await;
        }
    }

    async fn context(&self) -> Vec<Message> {
        self.inner.context().await
    }
}
