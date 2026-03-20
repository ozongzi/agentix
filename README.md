# agentix

[![crates.io](https://img.shields.io/crates/v/agentix.svg)](https://crates.io/crates/agentix)
[![docs.rs](https://img.shields.io/docsrs/agentix)](https://docs.rs/agentix)
[![license](https://img.shields.io/crates/l/agentix.svg)](https://github.com/ozongzi/agentix/blob/main/LICENSE-MIT)

A Rust framework for building LLM agents. Supports DeepSeek, OpenAI, Anthropic (Claude), and Google Gemini out of the box — plus any OpenAI-compatible endpoint. Define tools in plain Rust, plug them into an agent, and consume a stream of events as the model thinks, calls tools, and responds.

---

## Quickstart

Set your API key and add the dependency:

```bash
export DEEPSEEK_API_KEY="sk-..."
```

```toml
# Cargo.toml
[dependencies]
agentix = "0.1"
futures  = "0.3"
tokio    = { version = "1", features = ["full"] }
serde    = { version = "1", features = ["derive"] }
```

```rust
use agentix::{AgentEvent, DeepSeekAgent, tool};
use futures::StreamExt;
use serde_json::{Value, json};

struct Search;

#[tool]
impl agentix::Tool for Search {
    /// Search the web and return results.
    /// query: the search query
    async fn search(&self, query: String) -> Value {
        json!({ "results": format!("results for: {query}") })
    }
}

#[tokio::main]
async fn main() {
    let token = std::env::var("DEEPSEEK_API_KEY").unwrap();

    let mut stream = DeepSeekAgent::new(token)
        .with_tool(Search)
        .chat("What's the latest news about Rust?");

    while let Some(event) = stream.next().await {
        match event.unwrap() {
            AgentEvent::Token(text)       => print!("{text}"),
            AgentEvent::ToolCall(c)       => println!("\n[calling {}]", c.name),
            AgentEvent::ToolResult(r)     => println!("[result] {}", r.result),
            AgentEvent::ReasoningToken(t) => print!("{t}"),
        }
    }
}
```

The agent runs the full loop for you: it calls the model, dispatches any tool calls, feeds the results back, and keeps going until the model stops requesting tools.

---

## Defining tools

Annotate an `impl Tool for YourStruct` block with `#[tool]`. Each method becomes a callable tool:

- **Doc comment** on each method → tool description
- **`/// param: description`** lines → argument descriptions
- Return type just needs to be `serde::Serialize` — the macro handles the JSON schema

```rust
use agentix::tool;
use serde_json::{Value, json};

struct Calculator;

#[tool]
impl agentix::Tool for Calculator {
    /// Add two numbers together.
    /// a: first number
    /// b: second number
    async fn add(&self, a: f64, b: f64) -> Value {
        json!({ "result": a + b })
    }

    /// Multiply two numbers.
    /// a: first number
    /// b: second number
    async fn multiply(&self, a: f64, b: f64) -> Value {
        json!({ "result": a * b })
    }
}

// or just add it to an async fn

#[tool]
/// Divide two numbers.
/// a: first number
/// b: second number
async fn divide(&self, a: f64, b: f64) -> Value {
    json!({ "result": a / b })
}
```

One struct can have multiple methods — they register as separate tools. Stack as many tools as you need with `.with_tool(...)`.

---

## Streaming

Call `.streaming()` to get token-by-token output instead of waiting for the full response:

```rust
let mut stream = DeepSeekAgent::new(token)
    .streaming()
    .with_tool(Search)
    .chat("Search for something and summarise it");

while let Some(event) = stream.next().await {
    match event.unwrap() {
        AgentEvent::Token(t)      => { print!("{t}"); io::stdout().flush().ok(); }
        AgentEvent::ToolCall(c)   => {
            // In streaming mode, ToolCall fires once per SSE chunk.
            // First chunk: c.delta is empty, c.name is set — good moment to show "calling X".
            // Subsequent chunks: c.delta contains incremental argument JSON.
            // In non-streaming mode, exactly one ToolCall fires with the full args in c.delta.
            if c.delta.is_empty() { println!("\n[calling {}]", c.name); }
        }
        AgentEvent::ToolResult(r) => println!("[done] {}: {}", r.name, r.result),
        _                         => {}
    }
}
```

### AgentEvent reference

| Variant | When | Notes |
|---------|------|-------|
| `Token(String)` | Model is speaking | Streaming: one fragment per chunk. Non-streaming: whole reply at once. |
| `ReasoningToken(String)` | Model is thinking | Only from reasoning models (e.g. `deepseek-reasoner`). |
| `ToolCall(ToolCallChunk)` | Tool call in progress | `chunk.id`, `chunk.name`, `chunk.delta`. Streaming: multiple per call. Non-streaming: one per call. |
| `ToolResult(ToolCallResult)` | Tool finished | `result.name`, `result.args`, `result.result`. |

---

## Using a different model or provider

Four providers are built in, each with its own typed agent and correct wire format:

```rust
use agentix::{DeepSeekAgent, OpenAIAgent, AnthropicAgent, GeminiAgent};

// DeepSeek (default base URL: https://api.deepseek.com)
let agent = DeepSeekAgent::new(token);                          // deepseek-chat
let agent = DeepSeekAgent::new(token).with_model("deepseek-reasoner");

// DeepSeek via a custom endpoint (e.g. OpenRouter)
let agent = DeepSeekAgent::custom(
    "sk-or-...",
    "https://openrouter.ai/api/v1",
    "meta-llama/llama-3.3-70b-instruct:free",
);

// OpenAI — official API
let agent = OpenAIAgent::official(token, "gpt-4o");

// OpenAI — any compatible endpoint
let agent = OpenAIAgent::new(token, "https://my-proxy.example.com/v1", "gpt-4o");

// Anthropic (Claude) — official API
let agent = AnthropicAgent::official(token, "claude-sonnet-4-5");

// Anthropic — custom endpoint
let agent = AnthropicAgent::new(token, "https://api.anthropic.com", "claude-opus-4-5");

// Gemini — official API
let agent = GeminiAgent::official(token, "gemini-2.0-flash");

// Gemini — custom endpoint
let agent = GeminiAgent::new(
    token,
    "https://generativelanguage.googleapis.com/v1beta",
    "gemini-2.5-pro",
);
```

All four agent types share the same builder API (`.streaming()`, `.with_tool()`, `.with_system_prompt()`, etc.) and produce the same `AgentEvent` stream.

---

## Custom top-level request fields (`extra_body`)

The `extra_body` mechanism merges arbitrary top-level JSON fields into the HTTP request body. Useful for provider-specific or experimental options not modelled by the typed request structure.

Fields are flattened into the top-level JSON, so they appear as peers to `messages`, `model`, etc. Avoid colliding with those reserved keys.

```rust
use serde_json::json;
use agentix::DeepSeekAgent;

// Merge a map of fields
let agent = DeepSeekAgent::new(token)
    .extra_body({
        let mut m = serde_json::Map::new();
        m.insert("provider_option".to_string(), json!("value"));
        m
    });

// Or set a single field
let agent = DeepSeekAgent::new(token)
    .extra_field("provider_option", json!("value"));
```

---

## Injecting messages mid-run

Call `agent.interrupt_sender()` to get a channel sender that injects user messages into the running agent loop — useful when the user types something while tools are executing.

```rust
let agent = DeepSeekAgent::new(token)
    .streaming()
    .with_tool(SlowTool);

// Grab the sender before consuming the agent into a stream.
let tx = agent.interrupt_sender();

// In another task, send an interrupt at any time.
tokio::spawn(async move {
    tokio::time::sleep(Duration::from_secs(2)).await;
    tx.send("Actually, cancel that and do X instead.".into()).unwrap();
});

let mut stream = agent.chat("Do the slow thing.");
```

Behaviour:
- **Between turns**: queued interrupts are drained before the next API call.
- **During tool execution**: the running tool future is cancelled, a placeholder error result is recorded, and the injected message is appended to history before the next API turn.
- The sender is `tokio::sync::mpsc::UnboundedSender<String>` — cheap to clone, non-blocking.

---

## MCP tools

MCP (Model Context Protocol) lets you use external processes as tools — Node scripts, Python services, anything that speaks MCP over stdio:

```toml
[dependencies]
agentix = { version = "0.1", features = ["mcp"] }
```

```rust
use agentix::{DeepSeekAgent, McpTool};

let agent = DeepSeekAgent::new(token)
    .with_tool(McpTool::stdio("npx", &["-y", "@playwright/mcp"]).await?);
```

---

## Exposing tools as an MCP server

The `mcp-server` feature lets you turn any `ToolBundle` into a standalone MCP server so other LLM clients (Claude Desktop, MCP Studio, etc.) can call your Rust tools.

```toml
[dependencies]
agentix = { version = "0.1", features = ["mcp-server"] }
tokio   = { version = "1", features = ["full"] }
```

### Stdio mode (Claude Desktop / MCP Studio)

```rust
use agentix::{McpServer, ToolBundle, tool};

struct Calculator;

#[tool]
impl agentix::Tool for Calculator {
    /// Add two numbers.
    /// a: first operand
    /// b: second operand
    async fn add(&self, a: f64, b: f64) -> f64 { a + b }

    /// Multiply two numbers.
    /// a: first operand
    /// b: second operand
    async fn multiply(&self, a: f64, b: f64) -> f64 { a * b }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    McpServer::new(ToolBundle::new().with(Calculator))
        .with_name("my-calc-server")
        .serve_stdio()
        .await?;
    Ok(())
}
```

Register it in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "my-calc": {
      "command": "/path/to/your/binary"
    }
  }
}
```

### HTTP mode (Streamable HTTP transport)

```rust
use agentix::{McpServer, ToolBundle, tool};

struct Search;

#[tool]
impl agentix::Tool for Search {
    /// Search the web.
    /// query: what to search for
    async fn search(&self, query: String) -> String {
        format!("results for: {query}")
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    McpServer::new(ToolBundle::new().with(Search))
        .serve_http("0.0.0.0:3000")
        .await?;
    Ok(())
}
```

### Custom routing

For custom Axum routing, use `into_http_service()` to get a Tower-compatible service:

```rust
use agentix::{McpServer, ToolBundle};
use rmcp::transport::streamable_http_server::tower::StreamableHttpServerConfig;

let service = McpServer::new(ToolBundle::new().with(MyTools))
    .into_http_service(Default::default());

let router = axum::Router::new()
    .nest_service("/mcp", service)
    .route("/health", axum::routing::get(|| async { "ok" }));

let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
axum::serve(listener, router).await?;
```

---

## Tool Bundle

`ToolBundle` groups multiple `Tool` implementations and builds a name→index map for O(1) dispatch.

```rust
use agentix::{DeepSeekAgent, ToolBundle};

let tools = ToolBundle::new()
    .with(FileTools)
    .with(SearchTools)
    .with(ShellTools);

let agent = DeepSeekAgent::new(token)
    .with_tool(tools)
    .with_tool(UiTools { /* ... */ });
```

---

## Contributing

PRs welcome. Keep changes focused; update public API docs when behaviour changes.

## License

MIT OR Apache-2.0
