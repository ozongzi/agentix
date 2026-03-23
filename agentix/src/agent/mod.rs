pub mod node;

pub use node::AgentNode;

use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, OnceCell, RwLock, broadcast};
use futures::stream::{BoxStream, StreamExt};

use crate::client::LlmClient;
use crate::memory::{InMemory, Memory};
use crate::msg::{AgentInput, AgentEvent};
use crate::node::Node;
use crate::tool_trait::{Tool, ToolBundle};
use crate::types::UsageStats;
use crate::error::ApiError;

// ── Runtime ───────────────────────────────────────────────────────────────────

struct Runtime {
    tx:      mpsc::Sender<AgentInput>,
    /// Broadcast sender — subscribe anytime to get all future events.
    bcast:   broadcast::Sender<AgentEvent>,
    tools:   Arc<RwLock<ToolBundle>>,
}

// ── Agent ─────────────────────────────────────────────────────────────────────

/// A high-level agent handle.
///
/// # Usage
///
/// ```ignore
/// let mut agent = agentix::deepseek(api_key)
///     .system_prompt("You are helpful.")
///     .tool(MyTool);
///
/// // One-shot: lazy stream for this turn only (ends at Done).
/// let mut stream = agent.chat("hello").await?;
/// while let Some(ev) = stream.next().await { ... }
///
/// // Fire-and-forget send, then subscribe to the continuous output stream.
/// agent.send("follow-up").await?;
/// let rx = agent.subscribe();
/// while let Some(ev) = rx.next().await { ... }
///
/// // Add a tool even after the runtime has started.
/// agent.add_tool(AnotherTool).await;
/// ```
pub struct Agent {
    client:       Option<LlmClient>,
    memory:       Option<Box<dyn Memory + Send>>,
    staged_tools: Vec<Box<dyn Tool>>,
    usage:        Arc<Mutex<UsageStats>>,
    runtime:      OnceCell<Runtime>,
}

impl Agent {
    pub fn new(client: LlmClient) -> Self {
        Self {
            client:       Some(client),
            memory:       Some(Box::new(InMemory::new())),
            staged_tools: Vec::new(),
            usage:        Arc::new(Mutex::new(UsageStats::default())),
            runtime:      OnceCell::new(),
        }
    }

    // ── Builder methods ──────────────────────────────────────────────────────

    pub fn model(self, m: impl Into<String>) -> Self {
        if let Some(ref c) = self.client { c.model(m); }
        self
    }
    pub fn base_url(self, url: impl Into<String>) -> Self {
        if let Some(ref c) = self.client { c.base_url(url); }
        self
    }
    pub fn system_prompt(self, p: impl Into<String>) -> Self {
        if let Some(ref c) = self.client { c.system_prompt(p); }
        self
    }
    pub fn max_tokens(self, n: u32) -> Self {
        if let Some(ref c) = self.client { c.max_tokens(n); }
        self
    }
    pub fn temperature(self, t: f32) -> Self {
        if let Some(ref c) = self.client { c.temperature(t); }
        self
    }
    pub fn memory(mut self, m: impl Memory + 'static) -> Self {
        self.memory = Some(Box::new(m));
        self
    }

    /// Add a tool before the runtime starts (builder-style).
    pub fn tool(mut self, t: impl Tool + 'static) -> Self {
        self.staged_tools.push(Box::new(t));
        self
    }

    /// Add a tool at any time — before or after the first interaction.
    pub async fn add_tool(&mut self, t: impl Tool + 'static) {
        match self.runtime.get() {
            None => self.staged_tools.push(Box::new(t)),
            Some(rt) => rt.tools.write().await.push(t),
        }
    }

    // ── Runtime init ─────────────────────────────────────────────────────────

    fn start_runtime(&mut self) {
        let client = self.client.take().expect("client already consumed");
        let memory = self.memory.take().unwrap_or_else(|| Box::new(InMemory::new()));

        let mut bundle = ToolBundle::new();
        for t in self.staged_tools.drain(..) {
            bundle.push_boxed(t);
        }

        let tools_arc = Arc::new(RwLock::new(bundle));
        let (tx, rx) = mpsc::channel::<AgentInput>(64);
        let (bcast, _) = broadcast::channel::<AgentEvent>(256);

        let node = AgentNode::with_tools_arc(
            client,
            Arc::clone(&tools_arc),
            memory,
            Arc::clone(&self.usage),
        );

        let input_stream: BoxStream<'static, AgentInput> = async_stream::stream! {
            let mut rx = rx;
            while let Some(item) = rx.recv().await { yield item; }
        }.boxed();

        // Pump the AgentNode output into the broadcast channel.
        let bcast_tx = bcast.clone();
        let mut output = node.run(input_stream);
        tokio::spawn(async move {
            while let Some(ev) = output.next().await {
                // Ignore send errors — no subscribers is fine.
                let _ = bcast_tx.send(ev);
            }
        });

        let _ = self.runtime.set(Runtime { tx, bcast, tools: tools_arc });
    }

    fn ensure_runtime(&mut self) {
        if self.runtime.get().is_none() {
            self.start_runtime();
        }
    }

    fn rt(&mut self) -> &mut Runtime {
        self.ensure_runtime();
        unsafe { self.runtime.get_mut().unwrap_unchecked() }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Send a user message and return a **lazy** stream of [`AgentEvent`]s for
    /// this turn only.  The stream ends when [`AgentEvent::Done`] is emitted.
    pub async fn chat(&mut self, text: impl Into<String>) -> Result<BoxStream<'static, AgentEvent>, ApiError> {
        // Subscribe *before* sending to avoid missing early events.
        let mut rx = self.rt().bcast.subscribe();
        self.rt().tx.send(AgentInput::User(vec![text.into().into()])).await
            .map_err(|_| ApiError::Other("Agent runtime closed".into()))?;

        Ok(async_stream::stream! {
            loop {
                match rx.recv().await {
                    Ok(ev) => {
                        let done = matches!(ev, AgentEvent::Done);
                        yield ev;
                        if done { break; }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        yield AgentEvent::Error(format!("subscriber lagged, {} events dropped", n));
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }.boxed())
    }

    /// Send any input to the agent without waiting for a response.
    ///
    /// Accepts anything that converts to [`AgentInput`]:
    /// - `&str` / `String` — sends a user message
    /// - [`AgentInput`] directly — for tool results, abort signals, etc.
    ///
    /// Use [`Agent::subscribe`] to receive the resulting events.
    pub async fn send(&mut self, input: impl Into<AgentInput>) -> Result<(), ApiError> {
        self.rt().tx.send(input.into()).await
            .map_err(|_| ApiError::Other("Agent runtime closed".into()))
    }

    /// Subscribe to all future events as a continuous stream.
    /// Unlike [`Agent::chat`], this stream never stops at `Done`.
    pub fn subscribe(&mut self) -> BoxStream<'static, AgentEvent> {
        let mut rx = self.rt().bcast.subscribe();
        async_stream::stream! {
            loop {
                match rx.recv().await {
                    Ok(ev) => yield ev,
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        yield AgentEvent::Error(format!("subscriber lagged, {} events dropped", n));
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }.boxed()
    }

    /// Clone the sender so spawned tasks can send inputs concurrently.
    pub fn sender(&mut self) -> mpsc::Sender<AgentInput> {
        self.rt().tx.clone()
    }

    /// Abort the current generation.
    pub async fn abort(&mut self) -> Result<(), ApiError> {
        self.send(AgentInput::Abort).await
    }

    /// Accumulated token usage across all turns.
    pub fn usage(&self) -> UsageStats {
        self.usage.lock().unwrap().clone()
    }
}
