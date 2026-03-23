use std::sync::{Arc, Mutex};

use futures::StreamExt;
use serde_json::Value;
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, error, warn};

use crate::bus::EventBus;
use crate::client::LlmClient;
use crate::memory::{InMemory, Memory};
use crate::msg::Msg;
use crate::tool_trait::{Tool, ToolBundle};

// ── AgentInput ────────────────────────────────────────────────────────────────

enum AgentInput {
    Message(String),
    Abort,
}

// ── Runtime state ─────────────────────────────────────────────────────────────

enum RuntimeState {
    Pending {
        tools:  ToolBundle,
        memory: Box<dyn Memory + Send>,
        bus:    EventBus,
    },
    Running {
        inbox: mpsc::Sender<AgentInput>,
        bus:   EventBus,
    },
}

// ── AgentInner ────────────────────────────────────────────────────────────────

struct AgentInner {
    /// Always accessible — config changes work before and after first send().
    client: LlmClient,
    state:  Mutex<RuntimeState>,
    /// Lazily-created Sender<Msg> bridge for Node::input() compatibility.
    msg_inlet: std::sync::OnceLock<mpsc::Sender<Msg>>,
}

// ── Agent ─────────────────────────────────────────────────────────────────────

/// A clonable, actor-style agent handle.
///
/// Construction is **lazy**: the background task is spawned on the first call
/// to [`send`][Agent::send].  All configuration methods can be called freely
/// before and after that point — they take effect on the next API request.
///
/// All clones share the same inbox, event bus, and LLM config.
#[derive(Clone)]
pub struct Agent(Arc<AgentInner>);

impl Agent {
    // ── Constructors ──────────────────────────────────────────────────────────

    pub fn new(client: LlmClient) -> Self {
        Self::with_parts(client, ToolBundle::new(), Box::new(InMemory::new()), EventBus::new(512))
    }

    pub fn assemble(
        client: LlmClient,
        tools:  ToolBundle,
        memory: impl Memory + Send + 'static,
        bus:    EventBus,
    ) -> Self {
        Self::with_parts(client, tools, Box::new(memory), bus)
    }

    fn with_parts(
        client: LlmClient,
        tools:  ToolBundle,
        memory: Box<dyn Memory + Send>,
        bus:    EventBus,
    ) -> Self {
        Self(Arc::new(AgentInner {
            client,
            state: Mutex::new(RuntimeState::Pending { tools, memory, bus }),
            msg_inlet: std::sync::OnceLock::new(),
        }))
    }

    // ── Config methods — always available via LlmClient's RwLock ─────────────

    pub fn model(self, m: impl Into<String>) -> Self {
        self.0.client.model(m); self
    }

    pub fn base_url(self, url: impl Into<String>) -> Self {
        self.0.client.base_url(url); self
    }

    pub fn system_prompt(self, p: impl Into<String>) -> Self {
        self.0.client.system_prompt(p); self
    }

    pub fn max_tokens(self, n: u32) -> Self {
        self.0.client.max_tokens(n); self
    }

    pub fn temperature(self, t: f32) -> Self {
        self.0.client.temperature(t); self
    }

    /// Add a tool.  Must be called before the first [`send`][Self::send].
    pub fn tool(self, t: impl crate::tool_trait::Tool + 'static) -> Self {
        let mut s = self.0.state.lock().unwrap();
        if let RuntimeState::Pending { tools, .. } = &mut *s {
            tools.push(t);
        } else {
            warn!("tool() called after agent started — ignored");
        }
        drop(s);
        self
    }

    /// Replace the memory backend.  Must be called before the first [`send`][Self::send].
    pub fn memory(self, m: impl Memory + Send + 'static) -> Self {
        let mut s = self.0.state.lock().unwrap();
        if let RuntimeState::Pending { memory, .. } = &mut *s {
            *memory = Box::new(m);
        } else {
            warn!("memory() called after agent started — ignored");
        }
        drop(s);
        self
    }

    /// Replace the event bus.  Must be called before the first [`send`][Self::send].
    pub fn bus(self, b: EventBus) -> Self {
        let mut s = self.0.state.lock().unwrap();
        if let RuntimeState::Pending { bus, .. } = &mut *s {
            *bus = b;
        }
        drop(s);
        self
    }

    // ── Lazy spawn ────────────────────────────────────────────────────────────

    fn ensure_running(&self) -> (mpsc::Sender<AgentInput>, EventBus) {
        let mut s = self.0.state.lock().unwrap();

        if let RuntimeState::Running { inbox, bus } = &*s {
            return (inbox.clone(), bus.clone());
        }

        let (inbox_tx, inbox_rx) = mpsc::channel::<AgentInput>(64);

        // Swap out Pending to get owned config
        let prev = std::mem::replace(&mut *s, RuntimeState::Running {
            inbox: inbox_tx.clone(),
            bus:   EventBus::new(1), // placeholder, overwritten below
        });

        let (tools, memory, bus) = match prev {
            RuntimeState::Pending { tools, memory, bus } => (tools, memory, bus),
            RuntimeState::Running { .. } => unreachable!(),
        };

        *s = RuntimeState::Running { inbox: inbox_tx.clone(), bus: bus.clone() };
        drop(s);

        tokio::spawn(agent_loop(
            self.0.client.clone(), tools, memory, inbox_rx, bus.clone(),
        ));

        (inbox_tx, bus)
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Send a user message.  Spawns the background task on first call.
    pub async fn send(&self, msg: &str) {
        let (inbox, _) = self.ensure_running();
        if inbox.send(AgentInput::Message(msg.to_string())).await.is_err() {
            error!("agent inbox closed");
        }
    }

    /// Abort the current generation turn (if any).
    /// Queued messages are preserved and processed after the abort.
    pub async fn abort(&self) {
        let (inbox, _) = self.ensure_running();
        let _ = inbox.send(AgentInput::Abort).await;
    }

    /// Subscribe to all events emitted by this agent.
    pub fn subscribe(&self) -> broadcast::Receiver<Msg> {
        let (_, bus) = self.ensure_running();
        bus.subscribe()
    }

    /// Returns a `Sender<Msg>` that routes `Msg::User(text)` into the agent
    /// as a user message.  Other variants are silently ignored.
    ///
    /// The bridge task is created once and reused on subsequent calls.
    pub fn inbox_sender(&self) -> mpsc::Sender<Msg> {
        self.0.msg_inlet.get_or_init(|| {
            let (tx, mut rx) = mpsc::channel::<Msg>(64);
            let agent = self.clone();
            tokio::spawn(async move {
                while let Some(msg) = rx.recv().await {
                    if let Msg::User(text) = msg {
                        agent.send(&text).await;
                    }
                }
            });
            tx
        }).clone()
    }

    /// Access the underlying [`EventBus`].
    pub fn event_bus(&self) -> EventBus {
        let (_, bus) = self.ensure_running();
        bus
    }

    /// Read a snapshot of the current [`AgentConfig`].
    pub fn config_snapshot(&self) -> crate::config::AgentConfig {
        self.0.client.snapshot()
    }
}

// ── agent_loop ────────────────────────────────────────────────────────────────

async fn agent_loop(
    client:     LlmClient,
    tools:      ToolBundle,
    mut memory: Box<dyn Memory + Send>,
    mut inbox:  mpsc::Receiver<AgentInput>,
    bus:        EventBus,
) {
    // Wrap tools in Arc so concurrent tool-call futures can share it.
    let tools = Arc::new(tools);
    // Messages received while a generation is in-flight are buffered here.
    let mut queued: std::collections::VecDeque<String> = std::collections::VecDeque::new();

    loop {
        // ── Wait for next user message ────────────────────────────────────────
        let user_msg = if let Some(m) = queued.pop_front() {
            m
        } else {
            loop {
                match inbox.recv().await {
                    None                          => return, // sender dropped
                    Some(AgentInput::Message(m))  => break m,
                    Some(AgentInput::Abort)       => {}      // nothing to abort when idle
                }
            }
        };

        bus.send(Msg::TurnStart);
        bus.send(Msg::User(user_msg.clone()));
        memory.record(&Msg::User(user_msg)).await;

        // ── Tool-call rounds ──────────────────────────────────────────────────
        'turn: loop {
            let ctx  = memory.context().await;
            let defs = tools.raw_tools();

            let mut stream = match client.stream(&ctx, &defs).await {
                Ok(s)  => s,
                Err(e) => {
                    let msg = Msg::Error(e.to_string());
                    memory.record(&msg).await;
                    bus.send(msg);
                    break 'turn;
                }
            };

            // ── Drain stream, interruptible by Abort ──────────────────────────
            let mut pending_tool_calls: Vec<(String, String, String)> = vec![];

            'stream: loop {
                tokio::select! {
                    biased; // check inbox first to catch Abort promptly

                    maybe_input = inbox.recv() => {
                        match maybe_input {
                            None => return, // inbox closed
                            Some(AgentInput::Abort) => {
                                debug!("abort received mid-stream");
                                bus.send(Msg::Done);
                                memory.record(&Msg::Done).await;
                                break 'turn;
                            }
                            Some(AgentInput::Message(m)) => {
                                queued.push_back(m);
                            }
                        }
                    }

                    maybe_msg = stream.next() => {
                        match maybe_msg {
                            None => break 'stream, // stream ended unexpectedly
                            Some(Msg::Done) => {
                                if pending_tool_calls.is_empty() {
                                    bus.send(Msg::Done);
                                    memory.record(&Msg::Done).await;
                                    break 'turn;
                                }
                                break 'stream; // has tool calls — execute below
                            }
                            Some(msg @ Msg::ToolCall { .. }) => {
                                if let Msg::ToolCall { ref id, ref name, ref args } = msg {
                                    pending_tool_calls.push((
                                        id.clone(), name.clone(), args.clone(),
                                    ));
                                }
                                bus.send(msg);
                            }
                            Some(other) => {
                                memory.record(&other).await;
                                bus.send(other);
                            }
                        }
                    }
                }
            }

            if pending_tool_calls.is_empty() {
                break 'turn;
            }

            // ── Execute tool calls concurrently ───────────────────────────────
            let futs: Vec<_> = pending_tool_calls.into_iter().map(|(id, name, args)| {
                let tools = Arc::clone(&tools);
                async move {
                    let parsed: Value = serde_json::from_str(&args).unwrap_or(Value::Null);
                    let result = tools.call(&name, parsed).await;
                    (id, name, result)
                }
            }).collect();

            for (call_id, name, result) in futures::future::join_all(futs).await {
                let msg = Msg::ToolResult { call_id, name, result };
                memory.record(&msg).await;
                bus.send(msg);
            }
            // loop back → next LLM call with tool results in context
        }
    }
}
