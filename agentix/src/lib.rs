/*!
agentix — Multi-provider LLM client for Rust.

Supports DeepSeek, OpenAI, Anthropic, and Gemini out of the box.
The core API is a value-type [`Request`] that carries everything needed to
hit an LLM API — provider, credentials, model, messages, tools, and tuning.
Call [`Request::stream`] or [`Request::complete`] with a shared `reqwest::Client`.

# Quickstart

```no_run
use agentix::{Request, Provider, Message, UserContent, LlmEvent};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let http = reqwest::Client::new();

    let mut stream = Request::new(Provider::DeepSeek, std::env::var("DEEPSEEK_API_KEY")?)
        .system_prompt("You are helpful.")
        .user("Hello!")
        .stream(&http)
        .await?;

    while let Some(event) = stream.next().await {
        match event {
            LlmEvent::Token(t) => print!("{t}"),
            _                  => {}
        }
    }
    Ok(())
}
```
*/

pub(crate) mod config;
pub mod error;
pub mod msg;
pub(crate) mod provider;
pub mod raw;
pub mod request;
pub mod tool_trait;
pub mod types;

#[cfg(feature = "mcp")]
pub mod mcp;
#[cfg(feature = "mcp-server")]
pub mod mcp_server;
pub mod agent;

// ── Public API ────────────────────────────────────────────────────────────────

pub use error::ApiError;
pub use msg::LlmEvent;
pub use raw::shared::ToolDefinition;
pub use request::{
    ImageContent, ImageData, Message, Provider, Request, ResponseFormat,
    ToolCall, ToolChoice, UserContent, truncate_to_token_budget,
};
pub use types::{CompleteResponse, UsageStats};
pub use tool_trait::{Tool, ToolBundle, ToolOutput};
pub use agent::{AgentEvent, agent, agent_complete};

pub use agentix_macros::tool;
pub use schemars;
pub use serde;
pub use serde_json;
pub use async_trait;
pub use futures;

#[cfg(feature = "mcp")]
pub use mcp::McpTool;
#[cfg(feature = "mcp-server")]
pub use mcp_server::{McpServer, McpServerError, McpService};
