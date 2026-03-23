use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use futures::StreamExt;
use futures::stream::{self, BoxStream};

use crate::client::LlmClient;
use crate::memory::Memory;
use crate::msg::{AgentInput, AgentEvent};
use crate::node::{Node, LlmNode, ToolNode};
use crate::tool_trait::ToolBundle;
use crate::types::UsageStats;

// ── AgentNode (Internal stream transformer) ──────────────────────────────────

/// An internal node that transforms a stream of inputs into events.
/// Use this when building complex graph topologies.
#[derive(Clone)]
pub struct AgentNode {
    client: LlmClient,
    tools:  Arc<RwLock<ToolBundle>>,
    memory: Arc<Mutex<Box<dyn Memory + Send>>>,
    usage:  Arc<std::sync::Mutex<UsageStats>>,
}

impl AgentNode {
    pub fn new(client: LlmClient, tools: ToolBundle, memory: Box<dyn Memory + Send>, usage: Arc<std::sync::Mutex<UsageStats>>) -> Self {
        Self {
            client,
            tools: Arc::new(RwLock::new(tools)),
            memory: Arc::new(Mutex::new(memory)),
            usage,
        }
    }

    /// Construct with a pre-existing shared tool registry (for runtime tool insertion).
    pub(crate) fn with_tools_arc(client: LlmClient, tools: Arc<RwLock<ToolBundle>>, memory: Box<dyn Memory + Send>, usage: Arc<std::sync::Mutex<UsageStats>>) -> Self {
        Self {
            client,
            tools,
            memory: Arc::new(Mutex::new(memory)),
            usage,
        }
    }
}

impl Node for AgentNode {
    type Input = AgentInput;
    type Output = AgentEvent;

    fn run(self, mut input: BoxStream<'static, Self::Input>) -> BoxStream<'static, Self::Output> {
        let agent = self;

        async_stream::stream! {
            let mut pending_inputs = std::collections::VecDeque::new();

            loop {
                // 1. Determine next batch of inputs.
                let mut current_batch = Vec::new();
                
                let first = if let Some(item) = pending_inputs.pop_front() {
                    item
                } else {
                    match input.next().await {
                        Some(i) => i,
                        None => break,
                    }
                };
                current_batch.push(first);

                while let Some(i) = pending_inputs.pop_front() {
                    current_batch.push(i);
                }

                // 2. Process Batch
                let mut tool_results = Vec::new();
                let mut user_msgs = Vec::new();
                let mut aborted = false;

                for item in current_batch {
                    match item {
                        AgentInput::ToolResult { .. } => tool_results.push(item),
                        AgentInput::User(_)           => user_msgs.push(item),
                        AgentInput::Abort             => { aborted = true; break; }
                    }
                }
                if aborted { break; }

                for tr in tool_results { agent.memory.lock().await.record_input(&tr).await; }
                for um in user_msgs { agent.memory.lock().await.record_input(&um).await; }

                // 3. Interaction Loop
                'turn: loop {
                    let llm_node = LlmNode::new(
                        agent.client.clone(), 
                        Arc::clone(&agent.memory),
                        Some(Arc::clone(&agent.tools))
                    );
                    
                    let mut brain_out = llm_node.run(stream::iter(vec![None]).boxed());
                    let mut pending_tool_calls = Vec::new();

                    while let Some(ev) = brain_out.next().await {
                        match ev {
                            AgentEvent::ToolCall(ref tc) => {
                                pending_tool_calls.push(tc.clone());
                                yield ev;
                            }
                            AgentEvent::Usage(ref stats) => {
                                *agent.usage.lock().unwrap() += stats.clone();
                                yield ev;
                            }
                            AgentEvent::Done => {
                                if pending_tool_calls.is_empty() { yield ev; }
                            }
                            other => yield other,
                        }
                    }

                    if pending_tool_calls.is_empty() {
                        break 'turn; 
                    }

                    // 4. Execute Tools
                    let tool_node = ToolNode::new(Arc::clone(&agent.tools));
                    let mut hands_out = tool_node.run(stream::iter(pending_tool_calls).boxed());
                    let mut turn_aborted = false;
                    
                    let mut local_tool_results = Vec::new();

                    loop {
                        tokio::select! {
                            biased;
                            new_input = input.next() => {
                                match new_input {
                                    Some(AgentInput::Abort) => {
                                        yield AgentEvent::Done;
                                        turn_aborted = true;
                                        break;
                                    }
                                    Some(other) => { pending_inputs.push_back(other); }
                                    None => { turn_aborted = true; break; }
                                }
                            }
                            maybe_ev = hands_out.next() => {
                                if let Some(AgentEvent::ToolResult { ref call_id, ref result, .. }) = maybe_ev {
                                    local_tool_results.push(AgentInput::ToolResult { 
                                        call_id: call_id.clone(), 
                                        result: result.clone() 
                                    });
                                }
                                if let Some(ev) = maybe_ev {
                                    yield ev;
                                } else {
                                    break;
                                }
                            }
                        }
                    }

                    if turn_aborted { break 'turn; }
                    
                    for f in local_tool_results {
                        agent.memory.lock().await.record_input(&f).await;
                    }
                    
                    while let Some(i) = pending_inputs.pop_front() {
                        agent.memory.lock().await.record_input(&i).await;
                    }
                }
            }
        }.boxed()
    }
}
