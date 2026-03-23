/*!
agentix — Multi-provider LLM agent framework for Rust.

Supports DeepSeek, OpenAI, Anthropic, and Gemini out of the box.
Agents are actor-style handles; multiple agents wire together into a
[`Graph`] via typed [`Msg`] channels.

# Quickstart

```no_run
use agentix::Msg;

#[tokio::main]
async fn main() {
    let agent = agentix::deepseek(std::env::var("DEEPSEEK_API_KEY").unwrap())
        .system_prompt("You are helpful.")
        .max_tokens(1024);

    let mut rx = agent.subscribe();
    agent.send("Hello!").await;

    while let Ok(msg) = rx.recv().await {
        match msg {
            Msg::Token(t) => print!("{t}"),
            Msg::Done     => break,
            _             => {}
        }
    }
}
```

# Multi-agent pipeline

```no_run
use agentix::{Graph, Node, PromptTemplate, OutputParser, Msg};

#[tokio::main]
async fn main() {
    let prompt  = PromptTemplate::new("Score this review 0-10:\n{input}");
    let scorer  = agentix::deepseek(std::env::var("KEY").unwrap())
                    .system_prompt("Respond with only JSON: {\"score\": N}");
    let parser  = OutputParser::new(|s| {
        serde_json::from_str::<serde_json::Value>(&s)
            .ok()
            .and_then(|v| v["score"].as_i64().map(|n| n.to_string()))
            .unwrap_or_else(|| "0".into())
    });

    let _handle = Graph::new()
        .middleware(|msg| { eprintln!("[edge] {msg:?}"); Some(msg) })
        .edge(&prompt, &scorer)
        .edge(&scorer, &parser)
        .into_handle();

    prompt.input()
        .send(Msg::User(vec!["Great product, fast shipping!".into()]))
        .await
        .unwrap();
}
```

# Assembled vs streaming events

Every [`EventBus`] can be consumed two ways:

```no_run
# use agentix::Msg;
# use futures::StreamExt;
# #[tokio::main] async fn main() {
let agent = agentix::deepseek(std::env::var("KEY").unwrap());

// Raw — one Token per streaming chunk
let mut rx = agent.subscribe();

// Assembled — many Token chunks folded into one Token(full_text)
let mut stream = Box::pin(agent.event_bus().subscribe_assembled());
while let Some(msg) = stream.next().await {
    match msg {
        Msg::Token(full) => println!("{full}"),
        Msg::Done        => break,
        _                => {}
    }
}
# }
```
*/

pub mod bus;
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
mod agent;
pub mod context;
pub mod node;

#[cfg(feature = "mcp")]
pub mod mcp;
#[cfg(feature = "mcp-server")]
pub mod mcp_server;

// ── Public API ────────────────────────────────────────────────────────────────

pub use agent::Agent;
pub use bus::EventBus;
pub use client::LlmClient;
pub use config::AgentConfig;
pub use error::ApiError;
pub use memory::{InMemory, Memory, SlidingWindow};
pub use msg::{CustomMsg, Msg};
pub use provider::{AnthropicProvider, DeepSeekProvider, GeminiProvider, OpenAIProvider, Provider};
pub use request::{ImageContent, ImageData, Message, Request, ResponseFormat, ToolChoice, UserContent};
pub use context::SharedContext;
pub use node::{Graph, GraphHandle, MiddlewareFn, Node, OutputParser, PromptTemplate};
pub use tool_trait::{Tool, ToolBundle};

pub use agentix_macros::tool;
pub use schemars;
pub use serde;
pub use serde_json;
pub use async_trait;

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
