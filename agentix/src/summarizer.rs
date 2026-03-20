//! Conversation summarizer trait and built-in implementations.
//!
//! A [`Summarizer`] decides when and how to compress the agent's message history.
//!
//! | Type | Strategy |
//! |---|---|
//! | [`NoOpSummarizer`] | Never summarizes; used as the default. |
//! | [`SlidingWindowSummarizer`] | Keeps the last N messages; no API call. |
//! | [`LlmSummarizer`] | Calls the provider API to produce a condensed summary. |

use futures::Future;
use std::pin::Pin;

use crate::api::ApiClient;
use crate::error::ApiError;
use crate::request::{Message, Request, UserContent};
use crate::types::ProviderProtocol;

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Decides when and how to compress conversation history.
///
/// The trait is object-safe via `BoxFuture`.  Store it as
/// `Box<dyn Summarizer + Send + Sync>`.
pub trait Summarizer: Send + Sync {
    /// Return `true` if the history should be summarized before the next turn.
    fn should_summarize(&self, history: &[Message]) -> bool;

    /// Compress `history` in-place.
    fn summarize<'a>(
        &'a self,
        history: &'a mut Vec<Message>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ApiError>> + Send + 'a>>;
}

// ── NoOpSummarizer ────────────────────────────────────────────────────────────

/// Never summarizes.  This is the default for all agents.
#[derive(Debug, Clone, Default)]
pub struct NoOpSummarizer;

impl Summarizer for NoOpSummarizer {
    fn should_summarize(&self, _history: &[Message]) -> bool {
        false
    }

    fn summarize<'a>(
        &'a self,
        _history: &'a mut Vec<Message>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ApiError>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

// ── SlidingWindowSummarizer ───────────────────────────────────────────────────

/// Keeps only the most recent `window` messages, silently discarding older ones.
/// No API call is made.
///
/// # Example
///
/// ```no_run
/// use agentix::{DeepSeekAgent};
/// use agentix::summarizer::SlidingWindowSummarizer;
///
/// let agent = DeepSeekAgent::new("sk-...")
///     .with_summarizer(SlidingWindowSummarizer::new(20));
/// ```
#[derive(Debug, Clone)]
pub struct SlidingWindowSummarizer {
    /// Maximum messages to keep after trimming.
    pub window: usize,
    /// Trigger threshold; defaults to `window + 1`.
    pub trigger_at: Option<usize>,
}

impl SlidingWindowSummarizer {
    /// Create a summarizer that retains at most `window` messages.
    pub fn new(window: usize) -> Self {
        Self {
            window,
            trigger_at: None,
        }
    }

    /// Set the history length that triggers trimming (must be > `window`).
    pub fn trigger_at(mut self, n: usize) -> Self {
        self.trigger_at = Some(n.max(self.window + 1));
        self
    }
}

impl Summarizer for SlidingWindowSummarizer {
    fn should_summarize(&self, history: &[Message]) -> bool {
        let threshold = self.trigger_at.unwrap_or(self.window + 1);
        history.len() >= threshold
    }

    fn summarize<'a>(
        &'a self,
        history: &'a mut Vec<Message>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ApiError>> + Send + 'a>> {
        let window = self.window;
        Box::pin(async move {
            if history.len() <= window {
                return Ok(());
            }
            // Find the latest safe cut point: we may only cut at a boundary
            // where the message before the cut is a User message or a
            // tool-call-free Assistant message.  This prevents leaving
            // orphaned ToolResult messages without their preceding
            // Assistant{tool_calls} entry.
            let target_drop = history.len() - window;
            let mut cut = 0;
            for i in 0..target_drop {
                let safe = match &history[i] {
                    Message::User(_) => true,
                    Message::Assistant { tool_calls, .. } => tool_calls.is_empty(),
                    Message::ToolResult { .. } => false,
                };
                if safe {
                    cut = i + 1;
                }
            }
            if cut > 0 {
                history.drain(0..cut);
            }
            Ok(())
        })
    }
}

// ── LlmSummarizer ─────────────────────────────────────────────────────────────

/// Uses an LLM API call to compress the conversation history into a single
/// summary message, then replaces the old history with:
///
/// 1. A `User` message containing the condensed summary.
/// 2. The most recent `keep_last` messages (verbatim), so the model still has
///    immediate context.
///
/// The summarization request uses the same provider `P` as the parent agent.
///
/// # Example
///
/// ```no_run
/// use agentix::DeepSeekAgent;
/// use agentix::summarizer::LlmSummarizer;
/// use agentix::api::ApiClient;
/// use agentix::agent::agent_core::DeepSeek;
///
/// let client: ApiClient<DeepSeek> = ApiClient::new("sk-...");
/// let agent = DeepSeekAgent::new("sk-...")
///     .with_summarizer(
///         LlmSummarizer::new(client, "deepseek-chat")
///             .trigger_at(40)
///             .keep_last(10),
///     );
/// ```
pub struct LlmSummarizer<P: ProviderProtocol> {
    client: ApiClient<P>,
    model: String,
    /// History length that triggers summarization.
    trigger_at: usize,
    /// Number of most-recent messages to keep verbatim after summarization.
    keep_last: usize,
    /// Instruction sent to the model asking it to summarize.
    prompt: String,
}

impl<P: ProviderProtocol> LlmSummarizer<P> {
    pub fn new(client: ApiClient<P>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            trigger_at: 30,
            keep_last: 6,
            prompt: "Summarize the following conversation concisely, preserving all important facts, decisions, and context. Output only the summary text.".to_string(),
        }
    }

    /// Set the history length that triggers summarization (default: 30).
    pub fn trigger_at(mut self, n: usize) -> Self {
        self.trigger_at = n.max(self.keep_last + 1);
        self
    }

    /// Number of most-recent messages to keep verbatim (default: 6).
    pub fn keep_last(mut self, n: usize) -> Self {
        self.keep_last = n;
        self
    }

    /// Override the summarization instruction sent to the model.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }
}

impl<P: ProviderProtocol> Summarizer for LlmSummarizer<P> {
    fn should_summarize(&self, history: &[Message]) -> bool {
        history.len() >= self.trigger_at
    }

    fn summarize<'a>(
        &'a self,
        history: &'a mut Vec<Message>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ApiError>> + Send + 'a>> {
        Box::pin(async move {
            let keep = self.keep_last.min(history.len());
            let summarize_up_to = history.len().saturating_sub(keep);
            if summarize_up_to == 0 {
                return Ok(());
            }

            // Build a plain-text transcript of the messages to summarize.
            let mut transcript = String::new();
            for msg in &history[..summarize_up_to] {
                match msg {
                    Message::User(parts) => {
                        transcript.push_str("User: ");
                        for p in parts {
                            if let UserContent::Text(t) = p {
                                transcript.push_str(t);
                            }
                        }
                        transcript.push('\n');
                    }
                    Message::Assistant { content, .. } => {
                        if let Some(c) = content {
                            transcript.push_str("Assistant: ");
                            transcript.push_str(c);
                            transcript.push('\n');
                        }
                    }
                    Message::ToolResult { content, .. } => {
                        transcript.push_str("Tool result: ");
                        transcript.push_str(content);
                        transcript.push('\n');
                    }
                }
            }

            let req = Request {
                model: self.model.clone(),
                messages: vec![Message::User(vec![UserContent::Text(format!(
                    "{}\n\n<conversation>\n{}</conversation>",
                    self.prompt, transcript
                ))])],
                ..Default::default()
            };

            let resp = self.client.send(req).await?;
            let (events, _) = P::parse_response(resp);

            let summary_text: String = events
                .into_iter()
                .filter_map(|e| {
                    if let crate::types::AgentEvent::Token(t) = e {
                        Some(t)
                    } else {
                        None
                    }
                })
                .collect();

            // Replace [0..summarize_up_to] with one summary message, keep the rest.
            let tail: Vec<Message> = history.drain(summarize_up_to..).collect();
            history.clear();
            if !summary_text.trim().is_empty() {
                history.push(Message::User(vec![UserContent::Text(format!(
                    "[Conversation summary]\n{}",
                    summary_text.trim()
                ))]));
            }
            history.extend(tail);

            Ok(())
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::UserContent;

    fn user(text: &str) -> Message {
        Message::User(vec![UserContent::Text(text.to_string())])
    }

    #[tokio::test]
    async fn sliding_window_trims_to_window() {
        let mut history = vec![user("a"), user("b"), user("c"), user("d"), user("e")];
        let s = SlidingWindowSummarizer::new(2);
        assert!(s.should_summarize(&history));
        s.summarize(&mut history).await.unwrap();
        assert_eq!(history.len(), 2);
    }

    #[tokio::test]
    async fn sliding_window_noop_within_window() {
        let mut history = vec![user("a"), user("b")];
        let s = SlidingWindowSummarizer::new(4);
        assert!(!s.should_summarize(&history));
        s.summarize(&mut history).await.unwrap();
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn noop_never_triggers() {
        let history = vec![user("a"); 1000];
        assert!(!NoOpSummarizer.should_summarize(&history));
    }
}
