/*!
agentix — Multi-provider LLM client for Rust.

Supports DeepSeek, OpenAI, Anthropic, and Gemini out of the box.
The core API is a stateless [`LlmClient`] that turns `(messages, tools) → Stream<LlmEvent>`.
The caller owns the message history, tool dispatch, and control flow.

# Quickstart

```no_run
use agentix::{LlmClient, LlmEvent, Message, UserContent};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmClient::deepseek(std::env::var("DEEPSEEK_API_KEY")?);
    client.system_prompt("You are helpful.");

    let messages = vec![
        Message::User(vec![UserContent::Text("Hello!".into())]),
    ];
    let mut stream = client.stream(&messages, &[]).await?;

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

pub mod client;
pub mod config;
pub mod error;
pub mod msg;
pub mod provider;
pub mod raw;
pub mod request;
pub mod tool_trait;
pub mod types;

#[cfg(feature = "mcp")]
pub mod mcp;
#[cfg(feature = "mcp-server")]
pub mod mcp_server;

// ── Public API ────────────────────────────────────────────────────────────────

pub use client::LlmClient;
pub use config::AgentConfig;
pub use error::ApiError;
pub use msg::LlmEvent;
pub use provider::{AnthropicProvider, DeepSeekProvider, GeminiProvider, OpenAIProvider, Provider};
pub use raw::shared::ToolDefinition;
pub use request::{ImageContent, ImageData, Message, Request, ResponseFormat, ToolChoice, UserContent, ToolCall};
pub use types::{CompleteResponse, UsageStats};
pub use tool_trait::{Tool, ToolBundle, ToolOutput};

pub use agentix_macros::tool;
pub use schemars;
pub use serde;
pub use serde_json;
pub use async_trait;
pub use futures;

#[cfg(feature = "mcp")]
pub use mcp::McpTool;
#[cfg(feature = "mcp-server")]
pub use mcp_server::{McpServer, McpServerError};
