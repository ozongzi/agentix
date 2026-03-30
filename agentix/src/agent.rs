use async_stream::stream;
use futures::StreamExt;

use crate::msg::LlmEvent;
use crate::request::{Message, Request, ToolCall, truncate_to_token_budget};
use crate::tool_trait::{Tool, ToolOutput};
use crate::types::UsageStats;

// ── AgentEvent ────────────────────────────────────────────────────────────────

/// Events emitted by [`Agent::run`] over the course of a full generation loop
/// (potentially multiple LLM requests interleaved with tool executions).
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A text token from the LLM.
    Token(String),
    /// A reasoning/thinking token from the LLM.
    Reasoning(String),
    /// A streaming partial tool-call chunk (for live UI display).
    ToolCallChunk(crate::types::ToolCallChunk),
    /// A fully assembled tool call, about to be executed.
    ToolCallStart(ToolCall),
    /// Progress update from a tool (before the final result).
    ToolProgress { id: String, name: String, progress: String },
    /// The final result of a tool execution.
    ToolResult { id: String, name: String, content: String },
    /// Token usage from one LLM request.
    Usage(UsageStats),
    /// A recoverable stream error that was treated as end-of-stream.
    Warning(String),
    /// A fatal error — the stream will end after this.
    Error(String),
}

// ── Agent ─────────────────────────────────────────────────────────────────────

/// Stateless generation loop: given a base [`Request`] config and a
/// [`ToolBundle`], drives the LLM ↔ tool agentic loop and yields
/// [`AgentEvent`]s.
///
/// The caller is responsible for:
/// - Providing and mutating `history` (the conversation so far).
/// - Deciding when to abort (by dropping the stream).
/// - Persisting events to a database.
///
/// # Example
/// ```no_run
/// use agentix::{Agent, AgentEvent, Request, Provider, ToolBundle};
/// use futures::StreamExt;
///
/// # async fn run() {
/// let client = reqwest::Client::new();
/// let request = Request::new(Provider::OpenAI, "sk-...");
/// let tools = ToolBundle::default();
/// let agent = Agent::new(tools);
/// let history = vec![];
///
/// let mut stream = agent.run(client, request, history);
/// while let Some(event) = stream.next().await {
///     match event {
///         AgentEvent::Token(t) => print!("{t}"),
///         AgentEvent::ToolResult { name, content, .. } => {
///             println!("\n[{name}] → {content}");
///         }
///         AgentEvent::Error(e) => eprintln!("error: {e}"),
///         _ => {}
///     }
/// }
/// # }
/// ```
pub struct Agent {
    pub tools: std::sync::Arc<dyn Tool>,
    /// Token budget for history truncation before each LLM request (default: 25_000).
    /// Set to `usize::MAX` to disable truncation.
    pub token_budget: usize,
}

impl Agent {
    pub fn new(tools: impl Tool + 'static) -> Self {
        Self { tools: std::sync::Arc::new(tools), token_budget: 25_000 }
    }

    pub fn from_arc(tools: std::sync::Arc<dyn Tool>) -> Self {
        Self { tools, token_budget: 25_000 }
    }

    pub fn token_budget(mut self, budget: usize) -> Self {
        self.token_budget = budget;
        self
    }

    /// Run the agentic loop, yielding [`AgentEvent`]s.
    ///
    /// `history` starts with the messages to send and is extended with
    /// assistant messages and tool results as they are produced.
    /// The final history is available via [`AgentEvent::Done`] variant — but
    /// since there is no `Done` variant, the caller can inspect `history`
    /// after the stream ends (it is returned in `AgentEvent::Finished`).
    ///
    /// Takes ownership of `client` and `history`; both are moved into the
    /// returned stream so the stream is `'static`.
    pub fn run(
        &self,
        client: reqwest::Client,
        base_request: Request,
        mut history: Vec<Message>,
    ) -> futures::stream::BoxStream<'static, AgentEvent> {
        let tools = std::sync::Arc::clone(&self.tools);
        let tool_defs = self.tools.raw_tools();
        let token_budget = self.token_budget;

        Box::pin(stream! {
            loop {
                // ── Truncate history to token budget ──────────────────────
                truncate_to_token_budget(&mut history, token_budget);

                // ── Call LLM ──────────────────────────────────────────────
                let req = base_request.clone()
                    .messages(history.clone())
                    .tools(tool_defs.clone());

                let mut llm_stream = match req.stream(&client).await {
                    Ok(s) => s,
                    Err(e) => {
                        yield AgentEvent::Error(format!("LLM stream failed: {e}"));
                        return;
                    }
                };

                let mut reply_buf = String::new();
                let mut reasoning_buf = String::new();
                let mut tool_calls_buf: Vec<ToolCall> = Vec::new();

                // ── Consume LLM stream ────────────────────────────────────
                loop {
                    match llm_stream.next().await {
                        None | Some(LlmEvent::Done) => break,

                        Some(LlmEvent::Token(t)) => {
                            reply_buf.push_str(&t);
                            yield AgentEvent::Token(t);
                        }

                        Some(LlmEvent::Reasoning(t)) => {
                            reasoning_buf.push_str(&t);
                            yield AgentEvent::Reasoning(t);
                        }

                        Some(LlmEvent::ToolCallChunk(c)) => {
                            yield AgentEvent::ToolCallChunk(c);
                        }

                        Some(LlmEvent::ToolCall(tc)) => {
                            yield AgentEvent::ToolCallStart(tc.clone());
                            tool_calls_buf.push(tc);
                        }

                        Some(LlmEvent::Usage(u)) => {
                            yield AgentEvent::Usage(u);
                        }

                        Some(LlmEvent::Error(e)) => {
                            // Benign tail error (stream cut off after content arrived).
                            let benign = e.contains("Error in input stream")
                                && !reply_buf.trim().is_empty();
                            if benign {
                                yield AgentEvent::Warning(e);
                                break;
                            }
                            yield AgentEvent::Error(e);
                            return;
                        }
                    }
                }

                // ── Append assistant message to history ───────────────────
                let assistant_msg = Message::Assistant {
                    content: if reply_buf.is_empty() { None } else { Some(reply_buf.clone()) },
                    reasoning: if reasoning_buf.is_empty() { None } else { Some(reasoning_buf) },
                    tool_calls: tool_calls_buf.clone(),
                };
                if !reply_buf.is_empty() || !tool_calls_buf.is_empty() {
                    history.push(assistant_msg);
                }

                // ── No tool calls → generation complete ───────────────────
                if tool_calls_buf.is_empty() {
                    return;
                }

                // ── Execute tools ─────────────────────────────────────────
                for tc in &tool_calls_buf {
                    let args: serde_json::Value =
                        serde_json::from_str(&tc.arguments).unwrap_or(serde_json::json!({}));

                    yield AgentEvent::ToolProgress {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        progress: "executing...".into(),
                    };

                    let mut tool_stream = tools.call(&tc.name, args).await;
                    let mut result_val = serde_json::json!(null);

                    while let Some(output) = tool_stream.next().await {
                        match output {
                            ToolOutput::Progress(p) => {
                                yield AgentEvent::ToolProgress {
                                    id: tc.id.clone(),
                                    name: tc.name.clone(),
                                    progress: p,
                                };
                            }
                            ToolOutput::Result(v) => {
                                result_val = v;
                            }
                        }
                    }

                    let content = result_val.to_string();
                    yield AgentEvent::ToolResult {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        content: content.clone(),
                    };

                    history.push(Message::ToolResult {
                        call_id: tc.id.clone(),
                        content,
                    });
                }
                // Loop back → next LLM request with tool results appended.
            }
        })
    }
}
