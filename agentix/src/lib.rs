/*!
agentix — Multi-provider LLM agent framework for Rust

# Quickstart

## Simple streaming chat
```no_run
use agentix::{AgentEvent, DeepSeekAgent};
use futures::StreamExt;

#[tokio::main]
async fn main() {
    let token = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");
    let mut stream = DeepSeekAgent::new(token)
        .streaming()
        .chat("Hello from Rust!");
    while let Some(event) = stream.next().await {
        if let Ok(AgentEvent::Token(text)) = event {
            print!("{text}");
        }
    }
}
```

## Agent with a tool
```no_run
use agentix::{AgentEvent, DeepSeekAgent, tool};
use futures::StreamExt;
use serde_json::{Value, json};

struct EchoTool;

#[tool]
impl agentix::Tool for EchoTool {
    /// Echo the input back.
    /// input: the string to echo
    async fn echo(&self, input: String) -> Value {
        json!({ "echo": input })
    }
}

#[tokio::main]
async fn main() {
    let token = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");
    let mut stream = DeepSeekAgent::new(token).with_tool(EchoTool).chat("Please echo: hello");
    while let Some(event) = stream.next().await {
        match event {
            Err(e) => { eprintln!("Error: {e}"); break; }
            Ok(AgentEvent::Token(text))   => println!("Assistant: {text}"),
            Ok(AgentEvent::ToolCall(c))   => println!("calling {}({})", c.name, c.delta),
            Ok(AgentEvent::ToolResult(r)) => println!("result: {} -> {}", r.name, r.result),
            Ok(_) => {}
        }
    }
}
```

See the crate README for more details.
*/

pub mod agent;
pub mod api;
pub mod error;
#[cfg(feature = "mcp")]
pub mod mcp;
#[cfg(feature = "mcp-server")]
pub mod mcp_server;
pub mod raw;
pub mod request;
pub mod summarizer;
pub mod tool_trait;
pub mod types;

pub use agent::{
    AgentEvent, AnthropicAgent, DeepSeekAgent, GeminiAgent, OpenAIAgent, ToolCallChunk,
    ToolCallResult, ToolCommand,
};

pub use api::ApiClient;
pub use request::Message as ChatMessage;
pub use request::{Message, Request, ResponseFormat, ToolChoice};

pub use error::ApiError;
pub use summarizer::{LlmSummarizer, NoOpSummarizer, SlidingWindowSummarizer, Summarizer};

pub use tool_trait::Tool;
pub use tool_trait::ToolBundle;

pub use agentix_macros::tool;
pub use schemars;

#[cfg(feature = "mcp")]
pub use mcp::McpTool;

#[cfg(feature = "mcp-server")]
pub use mcp_server::{McpServer, McpServerError};
