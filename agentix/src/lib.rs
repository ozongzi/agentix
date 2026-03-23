/*!
agentix — Multi-provider LLM agent framework for Rust.

Supports DeepSeek, OpenAI, Anthropic, and Gemini out of the box.
Built on a pure stream-based architecture where [`Node`]s are stream transformers.

# Quickstart

```no_run
use agentix::{AgentInput, AgentEvent, Node};
use futures::StreamExt;

#[tokio::main]
async fn main() {
    let agent = agentix::deepseek(std::env::var("DEEPSEEK_API_KEY").unwrap())
        .system_prompt("You are helpful.");

    let input = futures::stream::iter(vec![
        AgentInput::User(vec!["Hello!".into()])
    ]).boxed();

    let mut response = agent.run(input);

    while let Some(event) = response.next().await {
        match event {
            AgentEvent::Token(t) => print!("{t}"),
            AgentEvent::Done     => break,
            _                    => {}
        }
    }
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
mod agent;
pub mod context;
pub mod node;

#[cfg(feature = "mcp")]
pub mod mcp;
#[cfg(feature = "mcp-server")]
pub mod mcp_server;

// ── Public API ────────────────────────────────────────────────────────────────

pub use agent::Agent;
pub use client::LlmClient;
pub use config::AgentConfig;
pub use error::ApiError;
pub use memory::{InMemory, Memory, SlidingWindow, TokenSlidingWindow, LlmSummarizer};
pub use msg::{CustomEvent, LlmEvent, AgentEvent, AgentInput};
pub use provider::{AnthropicProvider, DeepSeekProvider, GeminiProvider, OpenAIProvider, Provider};
pub use request::{ImageContent, ImageData, Message, Request, ResponseFormat, ToolChoice, UserContent, ToolCall};
pub use context::SharedContext;
pub use node::{Node, TapNode, PromptNode};
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
