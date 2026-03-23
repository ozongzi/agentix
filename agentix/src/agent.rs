use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use futures::StreamExt;
use futures::stream::BoxStream;
use tracing::debug;

use crate::client::LlmClient;
use crate::memory::{InMemory, Memory};
use crate::msg::{AgentInput, AgentEvent, LlmEvent};
use crate::node::Node;
use crate::tool_trait::Tool;
use crate::types::UsageStats;

// ── Agent ─────────────────────────────────────────────────────────────────────

/// A stream-based agent that transforms [`AgentInput`] into [`AgentEvent`].
#[derive(Clone)]
pub struct Agent {
    client: LlmClient,
    tools:  Arc<RwLock<crate::tool_trait::ToolBundle>>,
    memory: Arc<Mutex<Box<dyn Memory + Send>>>,
    usage:  Arc<std::sync::Mutex<UsageStats>>,
}

impl Agent {
    pub fn new(client: LlmClient) -> Self {
        let tools = crate::tool_trait::ToolBundle::new();
        let memory = InMemory::new();
        Self::assemble(client, tools, memory)
    }

    pub fn assemble(
        client: LlmClient,
        tools:  crate::tool_trait::ToolBundle,
        memory: impl Memory + Send + 'static,
    ) -> Self {
        Self {
            client,
            tools:  Arc::new(RwLock::new(tools)),
            memory: Arc::new(Mutex::new(Box::new(memory))),
            usage:  Arc::new(std::sync::Mutex::new(UsageStats::default())),
        }
    }

    pub fn model(self, m: impl Into<String>) -> Self {
        self.client.model(m); self
    }

    pub fn base_url(self, url: impl Into<String>) -> Self {
        self.client.base_url(url); self
    }

    pub fn system_prompt(self, p: impl Into<String>) -> Self {
        self.client.system_prompt(p); self
    }

    pub fn max_tokens(self, n: u32) -> Self {
        self.client.max_tokens(n); self
    }

    pub fn temperature(self, t: f32) -> Self {
        self.client.temperature(t); self
    }

    pub async fn tool(self, t: impl Tool + 'static) -> Self {
        self.tools.write().await.push(t);
        self
    }

    pub async fn memory(self, m: impl Memory + Send + 'static) -> Self {
        *self.memory.lock().await = Box::new(m);
        self
    }

    pub fn usage(&self) -> UsageStats {
        self.usage.lock().unwrap().clone()
    }

    pub fn config_snapshot(&self) -> crate::config::AgentConfig {
        self.client.snapshot()
    }
}

// ── Node implementation ───────────────────────────────────────────────────────

impl Node for Agent {
    type Input = AgentInput;
    type Output = AgentEvent;

    fn run(self, mut input: BoxStream<'static, Self::Input>) -> BoxStream<'static, Self::Output> {
        let agent = self;
        
        async_stream::stream! {
            let mut pending_inputs = std::collections::VecDeque::new();

            loop {
                // 1. Get next input (either from queue or from stream)
                let item = if let Some(item) = pending_inputs.pop_front() {
                    item
                } else {
                    match input.next().await {
                        Some(item) => item,
                        None => break, // input stream closed, terminate agent
                    }
                };

                // 2. Process Input
                let should_trigger_llm = match item {
                    AgentInput::User(_) | AgentInput::ToolResult { .. } => {
                        agent.memory.lock().await.record_input(&item).await;
                        true
                    }
                    AgentInput::Abort => false,
                };

                if !should_trigger_llm { continue; }

                // 3. Interaction Loop (Tool-call rounds)
                'turn: loop {
                    let ctx = agent.memory.lock().await.context().await;
                    let defs = agent.tools.read().await.raw_tools();

                    let mut stream: BoxStream<'static, LlmEvent> = match agent.client.stream(&ctx, &defs).await {
                        Ok(s) => s,
                        Err(e) => {
                            let ev = AgentEvent::Error(e.to_string());
                            agent.memory.lock().await.record_event(&ev).await;
                            yield ev;
                            break 'turn;
                        }
                    };

                    let mut pending_tool_calls = Vec::new();

                    // ── Read LLM Stream ─────────────────────────────────────────
                    'stream: loop {
                        tokio::select! {
                            biased;

                            new_input = input.next() => {
                                match new_input {
                                    Some(AgentInput::Abort) => {
                                        debug!("abort received mid-stream");
                                        let ev = AgentEvent::Done;
                                        agent.memory.lock().await.record_event(&ev).await;
                                        yield ev;
                                        break 'turn; // completely exit the turn
                                    }
                                    Some(other) => {
                                        pending_inputs.push_back(other);
                                    }
                                    None => break 'turn, // input closed
                                }
                            }

                            maybe_llm_ev = stream.next() => {
                                match maybe_llm_ev {
                                    Some(LlmEvent::Token(t)) => {
                                        let ev = AgentEvent::Token(t);
                                        agent.memory.lock().await.record_event(&ev).await;
                                        yield ev;
                                    }
                                    Some(LlmEvent::Reasoning(r)) => {
                                        let ev = AgentEvent::Reasoning(r);
                                        agent.memory.lock().await.record_event(&ev).await;
                                        yield ev;
                                    }
                                    Some(LlmEvent::ToolCallChunk(tc)) => {
                                        let ev = AgentEvent::ToolCallChunk(tc);
                                        agent.memory.lock().await.record_event(&ev).await;
                                        yield ev;
                                    }
                                    Some(LlmEvent::ToolCall(tc)) => {
                                        pending_tool_calls.push(tc.clone());
                                        let ev = AgentEvent::ToolCall(tc);
                                        agent.memory.lock().await.record_event(&ev).await;
                                        yield ev;
                                    }
                                    Some(LlmEvent::Usage(stats)) => {
                                        *agent.usage.lock().unwrap() += stats.clone();
                                        let ev = AgentEvent::Usage(stats);
                                        agent.memory.lock().await.record_event(&ev).await;
                                        yield ev;
                                    }
                                    Some(LlmEvent::Error(e)) => {
                                        let ev = AgentEvent::Error(e);
                                        agent.memory.lock().await.record_event(&ev).await;
                                        yield ev;
                                        break 'turn;
                                    }
                                    Some(LlmEvent::Done) | None => {
                                        break 'stream;
                                    }
                                }
                            }
                        }
                    }

                    if pending_tool_calls.is_empty() {
                        let ev = AgentEvent::Done;
                        agent.memory.lock().await.record_event(&ev).await;
                        yield ev;
                        break 'turn;
                    }

                    // ── Execute Tools (Concurrent & Interruptible) ─────────────
                    let mut tool_stream = futures::stream::iter(pending_tool_calls.into_iter().map(|tc| {
                        let tools = Arc::clone(&agent.tools);
                        async move {
                            let parsed: serde_json::Value = serde_json::from_str(&tc.arguments).unwrap_or(serde_json::Value::Null);
                            let result = tools.read().await.call(&tc.name, parsed).await;
                            (tc.id, tc.name, result)
                        }
                    })).buffer_unordered(10); // execute up to 10 tools concurrently

                    let mut aborted = false;
                    
                    loop {
                        tokio::select! {
                            biased;

                            new_input = input.next() => {
                                match new_input {
                                    Some(AgentInput::Abort) => {
                                        debug!("abort received during tool execution");
                                        let ev = AgentEvent::Done;
                                        agent.memory.lock().await.record_event(&ev).await;
                                        yield ev;
                                        aborted = true;
                                        break; // break tool loop
                                    }
                                    Some(other) => {
                                        pending_inputs.push_back(other);
                                    }
                                    None => {
                                        aborted = true;
                                        break;
                                    }
                                }
                            }

                            maybe_res = tool_stream.next() => {
                                match maybe_res {
                                    Some((call_id, name, result)) => {
                                        let ev = AgentEvent::ToolResult { call_id, name, result };
                                        agent.memory.lock().await.record_event(&ev).await;
                                        yield ev;
                                    }
                                    None => break, // all tools finished
                                }
                            }
                        }
                    }
                    
                    if aborted {
                        break 'turn;
                    }
                    
                    // Loop back to start the next LLM round with tool results
                }
            }
        }.boxed()
    }
}
