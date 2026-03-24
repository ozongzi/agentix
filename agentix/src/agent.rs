use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use tokio::sync::{mpsc, RwLock, broadcast};
use tokio::sync::Mutex as AsyncMutex;
use futures::stream::{BoxStream, StreamExt, FuturesUnordered};

use crate::client::LlmClient;
use crate::memory::{InMemory, Memory};
use crate::msg::{AgentInput, AgentEvent, LlmEvent};
use crate::request::ToolCall;
use crate::tool_trait::{Tool, ToolBundle, ToolOutput};
use crate::types::UsageStats;
use crate::error::ApiError;

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
    // lazy-init runtime
    tx:           Option<mpsc::Sender<AgentInput>>,
    bcast:        Option<broadcast::Sender<AgentEvent>>,
    tools:        Option<Arc<RwLock<ToolBundle>>>,
}

impl Agent {
    pub fn new(client: LlmClient) -> Self {
        Self {
            client:       Some(client),
            memory:       Some(Box::new(InMemory::new())),
            staged_tools: Vec::new(),
            usage:        Arc::new(Mutex::new(UsageStats::default())),
            tx:           None,
            bcast:        None,
            tools:        None,
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
        match self.tools.as_ref() {
            None => self.staged_tools.push(Box::new(t)),
            Some(arc) => arc.write().await.push(t),
        }
    }

    /// Replace the entire tool bundle at runtime.
    pub async fn replace_tool(&mut self, tools: ToolBundle) {
        let bundle = tools;
        match self.tools.as_ref() {
            None => {
                self.staged_tools.clear();
                self.staged_tools.push(Box::new(bundle));
            }
            Some(arc) => {
                *arc.write().await = bundle;
            }
        }
    }

    /// Remove all tools.
    pub async fn clear_tools(&mut self) {
        match self.tools.as_ref() {
            None => self.staged_tools.clear(),
            Some(arc) => arc.write().await.clear(),
        }
    }

    /// Remove a tool by function name.
    pub async fn delete_tool(&mut self, name: &str) {
        let names = [name.to_string()];
        match self.tools.as_ref() {
            None => {
                self.staged_tools.retain(|t| {
                    !t.raw_tools().iter().any(|r| r.function.name == name)
                });
            }
            Some(arc) => {
                arc.write().await.remove_by_names(&names);
            }
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

        let memory_arc = Arc::new(AsyncMutex::new(memory));
        let usage = Arc::clone(&self.usage);
        let bcast_tx = bcast.clone();
        let tools_for_task = Arc::clone(&tools_arc);

        tokio::spawn(agent_loop(rx, client, memory_arc, tools_for_task, usage, bcast_tx));

        self.tx    = Some(tx);
        self.bcast = Some(bcast);
        self.tools = Some(tools_arc);
    }

    fn ensure_runtime(&mut self) {
        if self.tx.is_none() {
            self.start_runtime();
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Send a user message and return a stream of events for this turn (ends at Done).
    pub async fn chat(&mut self, text: impl Into<String>) -> Result<BoxStream<'static, AgentEvent>, ApiError> {
        self.chat_multimodal(vec![text.into().into()]).await
    }

    /// Send a multimodal user message (text + images) and return a stream of events.
    pub async fn chat_multimodal(&mut self, parts: Vec<crate::request::UserContent>) -> Result<BoxStream<'static, AgentEvent>, ApiError> {
        self.ensure_runtime();
        let mut rx = self.bcast.as_ref().unwrap().subscribe();
        self.tx.as_ref().unwrap()
            .send(AgentInput::User(parts)).await
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

    /// Send any input without waiting for a response.
    pub async fn send(&mut self, input: impl Into<AgentInput>) -> Result<(), ApiError> {
        self.ensure_runtime();
        self.tx.as_ref().unwrap().send(input.into()).await
            .map_err(|_| ApiError::Other("Agent runtime closed".into()))
    }

    /// Subscribe to all future events as a continuous stream.
    pub fn subscribe(&mut self) -> BoxStream<'static, AgentEvent> {
        self.ensure_runtime();
        let mut rx = self.bcast.as_ref().unwrap().subscribe();
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
        self.ensure_runtime();
        self.tx.as_ref().unwrap().clone()
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

// ── Core agent loop (runs in a spawned task) ──────────────────────────────────

async fn agent_loop(
    mut rx: mpsc::Receiver<AgentInput>,
    client: LlmClient,
    memory: Arc<AsyncMutex<Box<dyn Memory + Send>>>,
    tools:  Arc<RwLock<ToolBundle>>,
    usage:  Arc<Mutex<UsageStats>>,
    bcast:  broadcast::Sender<AgentEvent>,
) {
    let mut pending: VecDeque<AgentInput> = VecDeque::new();
    loop {
        let Some(batch) = next_batch(&mut rx, &mut pending).await else { break };
        if !record_batch(batch, &memory).await { break; }
        'turn: loop {
            let tool_calls = call_llm(&client, &memory, &tools, &usage, &bcast).await;
            if tool_calls.is_empty() { break 'turn; }
            if !run_tools(tool_calls, &tools, &mut rx, &mut pending, &memory, &bcast).await {
                break 'turn;
            }
        }
    }
}

/// Pull the next batch of inputs: drain pending queue, or wait on rx.
/// Returns None when the channel is closed.
async fn next_batch(
    rx:      &mut mpsc::Receiver<AgentInput>,
    pending: &mut VecDeque<AgentInput>,
) -> Option<Vec<AgentInput>> {
    let first = if let Some(item) = pending.pop_front() {
        item
    } else {
        rx.recv().await?
    };
    let mut batch = vec![first];
    while let Some(i) = pending.pop_front() { batch.push(i); }
    Some(batch)
}

/// Record batch into memory. Returns false if Abort was received (caller should stop).
async fn record_batch(
    batch:  Vec<AgentInput>,
    memory: &Arc<AsyncMutex<Box<dyn Memory + Send>>>,
) -> bool {
    for item in batch {
        match item {
            AgentInput::Abort             => return false,
            AgentInput::ToolResult { .. } |
            AgentInput::User(_)           => memory.lock().await.record_input(&item).await,
        }
    }
    true
}

/// Call the LLM, emit events, return completed tool calls.
async fn call_llm(
    client: &LlmClient,
    memory: &Arc<AsyncMutex<Box<dyn Memory + Send>>>,
    tools:  &Arc<RwLock<ToolBundle>>,
    usage:  &Arc<Mutex<UsageStats>>,
    bcast:  &broadcast::Sender<AgentEvent>,
) -> Vec<ToolCall> {
    let ctx  = memory.lock().await.context().await;
    let defs = tools.read().await.raw_tools();
    let mut stream = match client.stream(&ctx, &defs).await {
        Ok(s)  => s,
        Err(e) => { let _ = bcast.send(AgentEvent::Error(e.to_string())); return vec![]; }
    };
    let mut tool_calls = Vec::new();
    while let Some(ev) = stream.next().await {
        let agent_ev = llm_to_agent(ev, &mut tool_calls, usage);
        let is_done  = matches!(agent_ev, AgentEvent::Done);
        memory.lock().await.record_event(&agent_ev).await;
        if !is_done || tool_calls.is_empty() { let _ = bcast.send(agent_ev); }
        if is_done { break; }
    }
    tool_calls
}

fn llm_to_agent(ev: LlmEvent, tool_calls: &mut Vec<ToolCall>, usage: &Arc<Mutex<UsageStats>>) -> AgentEvent {
    match ev {
        LlmEvent::Token(t)         => AgentEvent::Token(t),
        LlmEvent::Reasoning(r)     => AgentEvent::Reasoning(r),
        LlmEvent::ToolCallChunk(c) => AgentEvent::ToolCallChunk(c),
        LlmEvent::ToolCall(tc)     => { tool_calls.push(tc.clone()); AgentEvent::ToolCall(tc) }
        LlmEvent::Usage(u)         => { *usage.lock().unwrap() += u.clone(); AgentEvent::Usage(u) }
        LlmEvent::Error(e)         => AgentEvent::Error(e),
        LlmEvent::Done             => AgentEvent::Done,
    }
}

/// Execute tool calls concurrently, streaming results back.
/// Returns false if the turn was aborted.
async fn run_tools(
    tool_calls: Vec<ToolCall>,
    tools:      &Arc<RwLock<ToolBundle>>,
    rx:         &mut mpsc::Receiver<AgentInput>,
    pending:    &mut VecDeque<AgentInput>,
    memory:     &Arc<AsyncMutex<Box<dyn Memory + Send>>>,
    bcast:      &broadcast::Sender<AgentEvent>,
) -> bool {
    let mut futs: FuturesUnordered<_> = tool_calls.into_iter().map(|tc| {
        let tools_c = Arc::clone(tools);
        async move {
            let parsed = serde_json::from_str(&tc.arguments).unwrap_or(serde_json::Value::Null);
            let guard  = tools_c.read().await;
            let mut out = guard.call(&tc.name, parsed).await;
            let mut events = Vec::new();
            while let Some(output) = out.next().await {
                events.push(match output {
                    ToolOutput::Progress(p) => AgentEvent::ToolProgress { call_id: tc.id.clone(), name: tc.name.clone(), progress: p },
                    ToolOutput::Result(r)   => AgentEvent::ToolResult   { call_id: tc.id.clone(), name: tc.name.clone(), result: r },
                });
            }
            events
        }
    }).collect();

    let mut results_for_memory: Vec<AgentInput> = Vec::new();
    loop {
        tokio::select! {
            biased;
            new_input = rx.recv() => match new_input {
                Some(AgentInput::Abort) => { let _ = bcast.send(AgentEvent::Done); return false; }
                Some(other)             => pending.push_back(other),
                None                    => return false,
            },
            maybe = futs.next() => match maybe {
                Some(events) => {
                    for ev in events {
                        if let AgentEvent::ToolResult { ref call_id, ref result, .. } = ev {
                            results_for_memory.push(AgentInput::ToolResult { call_id: call_id.clone(), result: result.clone() });
                        }
                        let _ = bcast.send(ev);
                    }
                }
                None => break,
            },
        }
    }
    for tr in results_for_memory { memory.lock().await.record_input(&tr).await; }
    while let Some(i) = pending.pop_front() { memory.lock().await.record_input(&i).await; }
    true
}
