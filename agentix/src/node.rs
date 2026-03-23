use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use futures::stream::BoxStream;
use futures::StreamExt;

use crate::msg::{AgentInput, AgentEvent, LlmEvent};
use crate::context::SharedContext;
use crate::client::LlmClient;
use crate::memory::Memory;
use crate::tool_trait::{Tool, ToolBundle};

// ── Node trait ────────────────────────────────────────────────────────────────

/// The fundamental unit of composition in a stream-based architecture.
pub trait Node: Send + 'static {
    type Input:  Send + 'static;
    type Output: Send + 'static;

    fn run(self, input: BoxStream<'static, Self::Input>) -> BoxStream<'static, Self::Output>;
}

// ── LlmNode (The Brain) ──────────────────────────────────────────────────────

pub struct LlmNode {
    client: LlmClient,
    memory: Arc<Mutex<Box<dyn Memory + Send>>>,
    tools:  Option<Arc<RwLock<ToolBundle>>>,
}

impl LlmNode {
    pub fn new(
        client: LlmClient, 
        memory: Arc<Mutex<Box<dyn Memory + Send>>>,
        tools:  Option<Arc<RwLock<ToolBundle>>>,
    ) -> Self {
        Self { client, memory, tools }
    }
}

impl Node for LlmNode {
    type Input = Option<AgentInput>;
    type Output = AgentEvent;

    fn run(self, mut input: BoxStream<'static, Self::Input>) -> BoxStream<'static, Self::Output> {
        let client = self.client;
        let memory = self.memory;
        let tools_lock = self.tools;

        async_stream::stream! {
            while let Some(item_opt) = input.next().await {
                if let Some(item) = item_opt {
                    memory.lock().await.record_input(&item).await;
                    if let AgentInput::Abort = item { continue; }
                }

                let ctx = memory.lock().await.context().await;
                
                // Fetch tool definitions if a ToolBundle was provided
                let defs = if let Some(ref lock) = tools_lock {
                    lock.read().await.raw_tools()
                } else {
                    vec![]
                };

                let mut stream = match client.stream(&ctx, &defs).await {
                    Ok(s) => s,
                    Err(e) => {
                        yield AgentEvent::Error(e.to_string());
                        continue;
                    }
                };

                while let Some(llm_ev) = stream.next().await {
                    match llm_ev {
                        LlmEvent::Token(t) => {
                            let ev = AgentEvent::Token(t);
                            memory.lock().await.record_event(&ev).await;
                            yield ev;
                        }
                        LlmEvent::Reasoning(r) => {
                            let ev = AgentEvent::Reasoning(r);
                            memory.lock().await.record_event(&ev).await;
                            yield ev;
                        }
                        LlmEvent::ToolCallChunk(tc) => yield AgentEvent::ToolCallChunk(tc),
                        LlmEvent::ToolCall(tc) => {
                            let ev = AgentEvent::ToolCall(tc);
                            memory.lock().await.record_event(&ev).await;
                            yield ev;
                        }
                        LlmEvent::Usage(u) => yield AgentEvent::Usage(u),
                        LlmEvent::Error(e) => yield AgentEvent::Error(e),
                        LlmEvent::Done => {
                            let ev = AgentEvent::Done;
                            memory.lock().await.record_event(&ev).await;
                            yield ev;
                        }
                    }
                }
            }
        }.boxed()
    }
}

// ── ToolNode (The Hands) ──────────────────────────────────────────────────────

pub struct ToolNode {
    tools: Arc<RwLock<ToolBundle>>,
}

impl ToolNode {
    pub fn new(tools: Arc<RwLock<ToolBundle>>) -> Self {
        Self { tools }
    }
}

impl Node for ToolNode {
    type Input = crate::request::ToolCall;
    type Output = AgentEvent;

    fn run(self, mut input: BoxStream<'static, Self::Input>) -> BoxStream<'static, Self::Output> {
        let tools = self.tools;

        async_stream::stream! {
            // Collect all pending tool calls first, then execute them concurrently.
            let mut calls = Vec::new();
            while let Some(tc) = input.next().await {
                calls.push(tc);
            }

            use futures::stream::FuturesUnordered;
            use crate::tool_trait::ToolOutput;

            let mut futs = FuturesUnordered::new();
            for tc in calls {
                let tools_c = Arc::clone(&tools);
                futs.push(async move {
                    let parsed: serde_json::Value =
                        serde_json::from_str(&tc.arguments).unwrap_or(serde_json::Value::Null);
                    let mut out = tools_c.read().await.call(&tc.name, parsed).await;
                    let mut events: Vec<AgentEvent> = Vec::new();
                    while let Some(output) = out.next().await {
                        events.push(match output {
                            ToolOutput::Progress(p) => AgentEvent::ToolProgress {
                                call_id: tc.id.clone(),
                                name: tc.name.clone(),
                                progress: p,
                            },
                            ToolOutput::Result(r) => AgentEvent::ToolResult {
                                call_id: tc.id.clone(),
                                name: tc.name.clone(),
                                result: r,
                            },
                        });
                    }
                    events
                });
            }

            use futures::StreamExt as _;
            while let Some(events) = futs.next().await {
                for ev in events { yield ev; }
            }
        }.boxed()
    }
}

// ── Other utility nodes ───────────────────────────────────────────────────────

pub struct TapNode<I: Send + 'static> {
    callback: Box<dyn Fn(&I) + Send + Sync + 'static>,
}

impl<I: Send + 'static> TapNode<I> {
    pub fn new(f: impl Fn(&I) + Send + Sync + 'static) -> Self {
        Self { callback: Box::new(f) }
    }
}

impl<I: Send + 'static> Node for TapNode<I> {
    type Input = I;
    type Output = I;

    fn run(self, input: BoxStream<'static, Self::Input>) -> BoxStream<'static, Self::Output> {
        let cb = self.callback;
        input.map(move |item| {
            (cb)(&item);
            item
        }).boxed()
    }
}

pub struct PromptNode {
    template: String,
    context:  Option<SharedContext>,
}

impl PromptNode {
    pub fn new(template: impl Into<String>) -> Self {
        Self { template: template.into(), context: None }
    }

    pub fn context(mut self, ctx: SharedContext) -> Self {
        self.context = Some(ctx);
        self
    }
}

impl Node for PromptNode {
    type Input = String;
    type Output = AgentInput;

    fn run(self, input: BoxStream<'static, Self::Input>) -> BoxStream<'static, Self::Output> {
        let tmpl = self.template;
        let ctx = self.context;
        
        input.map(move |text| {
            let mut rendered = tmpl.replace("{input}", &text);
            if let Some(ref c) = ctx {
                for (k, _) in c.snapshot() {
                    let ph = format!("{{{k}}}");
                    if rendered.contains(&ph)
                        && let Some(val) = c.get_str(&k) {
                            rendered = rendered.replace(&ph, &val);
                        }
                }
            }
            AgentInput::User(vec![rendered.into()])
        }).boxed()
    }
}
