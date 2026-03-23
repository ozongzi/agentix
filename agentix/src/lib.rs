/*!
agentix — Multi-provider LLM agent framework for Rust.

Supports DeepSeek, OpenAI, Anthropic, and Gemini out of the box.
Built on a pure stream-based architecture where [`Node`]s are stream transformers.

# Quickstart

```no_run
use agentix::AgentEvent;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = agentix::deepseek(std::env::var("DEEPSEEK_API_KEY")?)
        .system_prompt("You are helpful.");

    let mut stream = agent.chat("Hello!").await?;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::Token(t) => print!("{t}"),
            _                    => {}
        }
    }
    Ok(())
}
```

# Agent API

All interaction goes through [`Agent`].  The runtime starts lazily on first use.

```no_run
use agentix::{AgentEvent, AgentInput};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = agentix::deepseek("sk-...")
        .system_prompt("You are helpful.")
        .max_tokens(1024);

    // chat() — lazy stream, ends at Done
    let mut stream = agent.chat("What is 2 + 2?").await?;
    while let Some(ev) = stream.next().await {
        if let AgentEvent::Token(t) = ev { print!("{t}"); }
    }

    // send() — fire-and-forget, accepts &str, String, or AgentInput
    agent.send("Follow up question").await?;
    agent.send(AgentInput::Abort).await?;          // abort

    // subscribe() — continuous stream, never stops at Done
    let mut rx = agent.subscribe();
    while let Some(ev) = rx.next().await { /* ... */ }

    // sender() — share the input channel with spawned tasks
    let tx = agent.sender();
    tokio::spawn(async move {
        tx.send(AgentInput::Abort).await.ok();
    });

    // add_tool() — add tools even after the first interaction
    agent.add_tool(agentix::tool_trait::ToolBundle::new()).await;

    // usage() — accumulated token counts across all turns
    println!("{:?}", agent.usage());
    Ok(())
}
```
*/

pub mod client;
pub mod config;
pub mod error;
pub mod markers;
pub mod memory;
pub mod msg;
pub mod provider;
pub mod raw;
pub mod request;
pub mod tool_trait;
pub mod types;
pub mod agent;
pub mod context;
pub mod node;

#[cfg(feature = "mcp")]
pub mod mcp;
#[cfg(feature = "mcp-server")]
pub mod mcp_server;

// ── Public API ────────────────────────────────────────────────────────────────

pub use agent::{Agent, AgentNode};
pub use client::LlmClient;
pub use config::AgentConfig;
pub use error::ApiError;
pub use memory::{InMemory, Memory, SlidingWindow, TokenSlidingWindow, LlmSummarizer};
pub use msg::{CustomEvent, LlmEvent, AgentEvent, AgentInput};
pub use provider::{AnthropicProvider, DeepSeekProvider, GeminiProvider, OpenAIProvider, Provider};
pub use request::{ImageContent, ImageData, Message, Request, ResponseFormat, ToolChoice, UserContent, ToolCall};
pub use context::SharedContext;
pub use node::{Node, TapNode, PromptNode};
pub use types::UsageStats;
pub use tool_trait::{Tool, ToolBundle, ToolOutput};

pub use agentix_macros::{tool, streaming_tool};
pub use schemars;
pub use serde;
pub use serde_json;
pub use async_trait;
pub use futures;

#[cfg(feature = "mcp")]
pub use mcp::McpTool;
#[cfg(feature = "mcp-server")]
pub use mcp_server::{McpServer, McpServerError};

// ── Free-function entry points ────────────────────────────────────────────────

/// Create an agent backed by the DeepSeek API. Default model: `deepseek-chat`.
pub fn deepseek(token: impl Into<String>) -> Agent {
    Agent::new(LlmClient::deepseek(token))
}

/// Create an agent backed by the OpenAI API (or any compatible endpoint). Default model: `gpt-4o`.
pub fn openai(token: impl Into<String>) -> Agent {
    Agent::new(LlmClient::openai(token))
}

/// Create an agent backed by the Anthropic Messages API. Default model: `claude-opus-4-5`.
pub fn anthropic(token: impl Into<String>) -> Agent {
    Agent::new(LlmClient::anthropic(token))
}

/// Create an agent backed by the Google Gemini API. Default model: `gemini-2.0-flash`.
pub fn gemini(token: impl Into<String>) -> Agent {
    Agent::new(LlmClient::gemini(token))
}
