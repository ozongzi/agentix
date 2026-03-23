use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;

use crate::client::LlmClient;
use crate::msg::{AgentEvent, AgentInput, LlmEvent};
use crate::request::{Message, Request};

/// A component that records conversation events and provides context.
#[async_trait]
pub trait Memory: Send {
    async fn record_input(&mut self, input: &AgentInput);
    async fn record_event(&mut self, event: &AgentEvent);

    /// Returns the conversation history used to build the next LLM request.
    async fn context(&self) -> Vec<Message>;
}

// ── InMemory ──────────────────────────────────────────────────────────────────

pub struct InMemory {
    messages: Vec<Message>,
    assistant_buf: String,
    reasoning_buf: String,
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

    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        self.messages = history;
        self
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn set_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }

    fn flush_assistant(&mut self) {
        let mut tool_calls: Vec<crate::request::ToolCall> = self
            .tool_call_bufs
            .drain()
            .map(|(id, (name, arguments))| crate::request::ToolCall {
                id,
                name,
                arguments,
            })
            .collect();

        if self.assistant_buf.is_empty() && self.reasoning_buf.is_empty() && tool_calls.is_empty() {
            return;
        }

        tool_calls.sort_by(|a, b| a.id.cmp(&b.id));

        let content = if self.assistant_buf.is_empty() {
            None
        } else {
            Some(self.assistant_buf.clone())
        };
        let reasoning = if self.reasoning_buf.is_empty() {
            None
        } else {
            Some(self.reasoning_buf.clone())
        };

        self.messages.push(Message::Assistant {
            content,
            reasoning,
            tool_calls,
        });

        self.assistant_buf.clear();
        self.reasoning_buf.clear();
    }
}

#[async_trait]
impl Memory for InMemory {
    async fn record_input(&mut self, input: &AgentInput) {
        match input {
            AgentInput::User(parts) => {
                self.messages.push(Message::User(parts.clone()));
            }
            AgentInput::ToolResult { call_id, result } => {
                self.flush_assistant();
                self.messages.push(Message::ToolResult {
                    call_id: call_id.clone(),
                    content: result.to_string(),
                });
            }
            AgentInput::Abort => {}
        }
    }

    async fn record_event(&mut self, event: &AgentEvent) {
        match event {
            AgentEvent::Token(t) => {
                self.assistant_buf.push_str(t);
            }
            AgentEvent::Reasoning(t) => {
                self.reasoning_buf.push_str(t);
            }
            AgentEvent::ToolCallChunk(_) => {
                // Chunks are for streaming display, we only record the fully assembled ToolCall.
            }
            AgentEvent::ToolCall(tc) => {
                self.tool_call_bufs
                    .entry(tc.id.clone())
                    .or_insert_with(|| (tc.name.clone(), String::new()))
                    .1
                    .push_str(&tc.arguments);
            }
            AgentEvent::ToolResult {
                call_id, result, ..
            } => {
                self.flush_assistant();
                self.messages.push(Message::ToolResult {
                    call_id: call_id.clone(),
                    content: result.to_string(),
                });
            }
            AgentEvent::Done => {
                self.flush_assistant();
            }
            _ => {}
        }
    }

    async fn context(&self) -> Vec<Message> {
        self.messages.clone()
    }
}

// ── SlidingWindow ─────────────────────────────────────────────────────────────

pub struct SlidingWindow {
    inner: InMemory,
    max: usize,
}

impl SlidingWindow {
    pub fn new(max: usize) -> Self {
        Self {
            inner: InMemory::new(),
            max,
        }
    }
}

#[async_trait]
impl Memory for SlidingWindow {
    async fn record_input(&mut self, input: &AgentInput) {
        self.inner.record_input(input).await;
    }
    async fn record_event(&mut self, event: &AgentEvent) {
        self.inner.record_event(event).await;
    }

    async fn context(&self) -> Vec<Message> {
        let all = self.inner.context().await;
        let len = all.len();
        if len <= self.max {
            return all;
        }

        // Trim from the front, but never split a tool_call / tool_result pair.
        let mut start = len - self.max;

        // Walk forward from `start` until we land on a safe boundary:
        // - not in the middle of an assistant message that has tool_calls
        //   whose results appear after it
        while start < len {
            match &all[start] {
                // Tool results must always be preceded by the assistant message
                // that requested them — skip past any orphaned tool results.
                Message::ToolResult { .. } => { start += 1; }
                Message::Assistant { tool_calls, .. } if !tool_calls.is_empty() => {
                    // If we start on an assistant msg with tool calls we'd keep
                    // the calls but potentially not their results — skip it.
                    start += 1;
                }
                _ => break,
            }
        }

        if start >= len {
            // Degenerate: window too small to hold a safe slice — return all.
            all
        } else {
            all[start..].to_vec()
        }
    }
}

// ── TokenSlidingWindow ────────────────────────────────────────────────────────

pub struct TokenSlidingWindow {
    inner: InMemory,
    max_tokens: usize,
}

impl TokenSlidingWindow {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            inner: InMemory::new(),
            max_tokens,
        }
    }
}

#[async_trait]
impl Memory for TokenSlidingWindow {
    async fn record_input(&mut self, input: &AgentInput) {
        self.inner.record_input(input).await;
    }
    async fn record_event(&mut self, event: &AgentEvent) {
        self.inner.record_event(event).await;
    }

    async fn context(&self) -> Vec<Message> {
        let all = self.inner.context().await;
        let mut total = 0;
        let mut result = Vec::new();

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

pub struct LlmSummarizer {
    inner: InMemory,
    client: LlmClient,
    trigger_at_tokens: usize,
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

    async fn summarize_if_needed(&mut self) {
        let messages = self.inner.messages();
        if messages.len() <= self.keep_recent {
            return;
        }

        let total_tokens: usize = messages.iter().map(|m| m.estimate_tokens()).sum();
        if total_tokens < self.trigger_at_tokens {
            return;
        }

        let to_summarize = &messages[..messages.len() - self.keep_recent];
        let recent = &messages[messages.len() - self.keep_recent..];

        let summary_req = Request { messages: to_summarize.to_vec(), ..Default::default() };

        let mut stream: BoxStream<'static, LlmEvent> =
            match self.client.stream(&summary_req.messages, &[]).await {
                Ok(s) => s,
                Err(_) => return,
            };

        let mut summary_text = String::new();
        while let Some(msg) = stream.next().await {
            match msg {
                LlmEvent::Token(t) => summary_text.push_str(&t),
                LlmEvent::Done => break,
                _ => {}
            }
        }

        if !summary_text.is_empty() {
            let mut new_history = Vec::new();
            // Use a user/assistant pair so providers that require alternating roles
            // (e.g. Anthropic, Gemini) don't reject the history as malformed.
            new_history.push(Message::User(vec![
                "[conversation summary request]".into(),
            ]));
            new_history.push(Message::Assistant {
                content: Some(format!("[auto-summary] {}", summary_text)),
                reasoning: None,
                tool_calls: vec![],
            });
            new_history.extend_from_slice(recent);
            self.inner.set_messages(new_history);
        }
    }
}

#[async_trait]
impl Memory for LlmSummarizer {
    async fn record_input(&mut self, input: &AgentInput) {
        self.inner.record_input(input).await;
    }
    async fn record_event(&mut self, event: &AgentEvent) {
        self.inner.record_event(event).await;
        if matches!(event, AgentEvent::Done) {
            self.summarize_if_needed().await;
        }
    }

    async fn context(&self) -> Vec<Message> {
        self.inner.context().await
    }
}
