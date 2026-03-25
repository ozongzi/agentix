# agentix

Stateless, multi-provider LLM client for Rust — streaming, non-streaming, tool calls, and MCP support.

DeepSeek · OpenAI · Anthropic · Gemini — one unified API.

## Installation

```toml
[dependencies]
agentix = "0.7"

# Optional: Model Context Protocol (MCP) tool support
# agentix = { version = "0.7", features = ["mcp"] }
```

---

## Quick Start

```rust
use agentix::{LlmClient, LlmEvent, Message, UserContent};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmClient::deepseek(std::env::var("DEEPSEEK_API_KEY")?);
    client.system_prompt("You are a helpful assistant.");

    let messages = vec![
        Message::User(vec![UserContent::Text("What is the capital of France?".into())]),
    ];
    let mut stream = client.stream(&messages, &[]).await?;

    while let Some(event) = stream.next().await {
        match event {
            LlmEvent::Token(t) => print!("{t}"),
            LlmEvent::Done     => break,
            _ => {}
        }
    }
    println!();
    Ok(())
}
```

---

## Providers

Four built-in providers, all using the same API:

```rust
use agentix::LlmClient;

// DeepSeek  (default model: deepseek-chat)
let client = LlmClient::deepseek("sk-...");

// OpenAI  (default model: gpt-4o)
let client = LlmClient::openai("sk-...");

// Anthropic / Claude  (default model: claude-opus-4-5)
let client = LlmClient::anthropic("sk-ant-...");

// Gemini  (default model: gemini-2.0-flash)
let client = LlmClient::gemini("AIza...");

// Any OpenAI-compatible endpoint (e.g. OpenRouter)
let client = LlmClient::openai("sk-or-...");
client.base_url("https://openrouter.ai/api/v1");
client.model("openrouter/free");

// From config strings (useful for dynamic provider selection)
let client = LlmClient::from_parts("openai", "sk-...", "https://api.openai.com/v1", "gpt-4o");
```

---

## LlmClient API

[`LlmClient`] is stateless — the caller owns message history and tool dispatch.
All clones share the same config and HTTP connection pool.

### `stream()` — streaming completion

```rust
let mut stream = client.stream(&messages, &tool_defs).await?;
while let Some(event) = stream.next().await {
    match event {
        LlmEvent::Token(t)         => print!("{t}"),
        LlmEvent::Reasoning(r)     => print!("[think] {r}"),
        LlmEvent::ToolCall(tc)     => println!("tool: {}({})", tc.name, tc.arguments),
        LlmEvent::Usage(u)         => println!("tokens: {}", u.total_tokens),
        LlmEvent::Error(e)         => eprintln!("error: {e}"),
        LlmEvent::Done             => break,
        _                          => {}
    }
}
```

### `complete()` — non-streaming completion

```rust
let resp = client.complete(&messages, &[]).await?;
println!("{}", resp.content.unwrap_or_default());
println!("reasoning: {:?}", resp.reasoning);
println!("tool_calls: {:?}", resp.tool_calls);
println!("usage: {:?}", resp.usage);
```

### Configuration

```rust
client.model("deepseek-reasoner");
client.base_url("https://custom.api/v1");
client.system_prompt("You are helpful.");
client.max_tokens(4096);
client.temperature(0.7);

// Read current config
let snap = client.snapshot();
```

---

## LlmEvent (what you receive from `stream()`)

- `Token(String)` — incremental response text
- `Reasoning(String)` — thinking/reasoning trace (e.g. DeepSeek-R1)
- `ToolCallChunk(ToolCallChunk)` — partial tool call for real-time UI
- `ToolCall(ToolCall)` — completed tool call
- `Usage(UsageStats)` — token usage for the turn
- `Done` — stream ended
- `Error(String)` — provider error

---

## Defining Tools

Annotate `impl Tool for YourStruct` with `#[tool]`. Each method becomes a callable tool.

```rust
use agentix::tool;
use serde_json::{json, Value};

struct Calculator;

#[tool]
impl agentix::Tool for Calculator {
    /// Add two numbers.
    /// a: first number
    /// b: second number
    async fn add(&self, a: i64, b: i64) -> i64 {
        a + b
    }

    /// Divide a by b.
    async fn divide(&self, a: f64, b: f64) -> Result<f64, String> {
        if b == 0.0 { Err("division by zero".into()) } else { Ok(a / b) }
    }
}
```

- Doc comment → tool description
- `/// param: description` lines → argument descriptions
- `Result::Err` automatically propagates as `{"error": "..."}` to the LLM

### Streaming tools

Add `#[streaming]` to yield `ToolOutput::Progress` / `ToolOutput::Result` incrementally:

```rust
use agentix::{tool, ToolOutput};

struct ProgressTool;

#[tool]
impl agentix::Tool for ProgressTool {
    /// Run a long job and stream progress.
    /// steps: number of steps
    #[streaming]
    fn long_job(&self, steps: u32) {
        async_stream::stream! {
            for i in 1..=steps {
                yield ToolOutput::Progress(format!("{i}/{steps}"));
            }
            yield ToolOutput::Result(serde_json::json!({ "done": true }));
        }
    }
}
```

Normal and streaming methods can be freely mixed in the same `#[tool]` block.

---

## MCP Tools

Use external processes as tools via the Model Context Protocol:

```rust
use agentix::McpTool;
use std::time::Duration;

let tool = McpTool::stdio("npx", &["-y", "@playwright/mcp"]).await?
    .with_timeout(Duration::from_secs(60));

// Add to a ToolBundle alongside regular tools
let mut bundle = agentix::ToolBundle::new();
bundle.push(tool);
```

---

## Reliability

- **Automatic retries** — exponential backoff for 429 / 5xx responses
- **HTTP timeouts** — 10 s connect, 120 s response (overridable via `LlmClient::with_http`)
- **Usage tracking** — per-request token accounting across all providers

---

## Changelog

### 0.7.0

- **Removed `Agent` struct** — `LlmClient` is now the sole entry point; callers own the loop
- **Removed `Memory` trait** — `InMemory`, `SlidingWindow`, `TokenSlidingWindow`, `LlmSummarizer` removed
- **Removed `AgentEvent` / `AgentInput`** — only `LlmEvent` remains
- **New `LlmClient::complete()`** — native non-streaming API for all four providers
- **New `CompleteResponse`** — content, reasoning, tool_calls, usage in one struct

### 0.6.0

- Non-streaming `complete()` method on `Provider` trait
- `post_json` helper for non-streaming HTTP POST with retry
- `CompleteResponse` type

### 0.5.0

- `Agent` API with `chat()`, `send()`, `subscribe()`, `add_tool()`, `abort()`, `usage()`
- Concurrent tool execution via `FuturesUnordered`
- `SlidingWindow` fix for orphaned tool messages
- Default HTTP timeouts (10 s connect, 120 s response)

### 0.4.x

- Initial multi-turn API
- DeepSeek, OpenAI, Anthropic, Gemini providers
- `#[tool]` and `#[streaming]` macros
- Memory backends, MCP tool support

---

## License

MIT OR Apache-2.0
