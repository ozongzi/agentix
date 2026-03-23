# agentix

[![crates.io](https://img.shields.io/crates/v/agentix.svg)](https://crates.io/crates/agentix)
[![docs.rs](https://img.shields.io/docsrs/agentix)](https://docs.rs/agentix)
[![license](https://img.shields.io/crates/l/agentix.svg)](https://github.com/ozongzi/agentix/blob/main/LICENSE-MIT)

A Rust framework for building LLM agents and multi-agent pipelines. Supports DeepSeek, OpenAI, Anthropic (Claude), and Google Gemini out of the box — plus any OpenAI-compatible endpoint.

Agents are **actor-style**: send a message, observe a stream of events. Multiple agents wire together into a [`Graph`](#graph--multi-agent-pipelines) via typed channels.

---

## Quickstart

```bash
export DEEPSEEK_API_KEY="sk-..."
```

```toml
[dependencies]
agentix = "0.3"
tokio   = { version = "1", features = ["full"] }
```

```rust
use agentix::Msg;

#[tokio::main]
async fn main() {
    let agent = agentix::deepseek(std::env::var("DEEPSEEK_API_KEY").unwrap())
        .system_prompt("You are a helpful assistant.")
        .max_tokens(1024);

    let mut rx = agent.subscribe();
    agent.send("What is the capital of France?").await;

    while let Ok(msg) = rx.recv().await {
        match msg {
            Msg::Token(t) => print!("{t}"),
            Msg::Done     => break,
            _             => {}
        }
    }
    println!();
}
```

---

## Providers

Four built-in providers, all using the same builder API:

```rust
// DeepSeek  (default model: deepseek-chat)
let agent = agentix::deepseek("sk-...")
    .model("deepseek-reasoner");

// OpenAI  (default model: gpt-4o)
let agent = agentix::openai("sk-...");

// Anthropic / Claude  (default model: claude-opus-4-5)
let agent = agentix::anthropic("sk-ant-...");

// Gemini  (default model: gemini-2.0-flash)
let agent = agentix::gemini("AIza...");

// Any OpenAI-compatible endpoint
use agentix::{Agent, LlmClient};
let agent = Agent::new(LlmClient::openai_compatible(
    "sk-...",
    "https://openrouter.ai/api/v1",
    "meta-llama/llama-3.3-70b-instruct:free",
));
```

---

## Builder chain

All configuration methods return `Self`, so the whole setup is one expression:

```rust
let agent = agentix::deepseek("sk-...")
    .model("deepseek-chat")
    .system_prompt("You are a code reviewer.")
    .temperature(0.2)
    .max_tokens(4096)
    .tool(MyTool)
    .memory(agentix::SlidingWindow::new(20));
```

---

## Msg — the event type

Every event that flows through an [`EventBus`] is a `Msg`:

| Variant | When |
|---------|------|
| `TurnStart` | Generation turn begins |
| `Done` | Turn (including all tool rounds) complete |
| `User(Vec<UserContent>)` | User message submitted — text and/or images |
| `Token(String)` | LLM output — one chunk in streaming, full text in assembled view |
| `Reasoning(String)` | Reasoning trace (e.g. DeepSeek-R1) — same streaming/assembled duality |
| `ToolCall { id, name, args }` | Complete tool invocation request |
| `ToolResult { call_id, name, result }` | Tool execution result |
| `Error(String)` | Error during generation |
| `Custom(Arc<dyn CustomMsg>)` | Application-defined payload |

### Streaming vs assembled

Subscribe to an [`EventBus`] in two ways:

```rust
// Raw streaming — Token arrives as individual chunks
let mut rx = agent.subscribe();          // broadcast::Receiver<Msg>

// Assembled — Token chunks folded into one Token(full_text) before Done
let stream = agent.event_bus().subscribe_assembled();  // impl Stream<Item = Msg>
```

The assembled view looks identical to what a non-streaming provider emits — same variant names, just complete content.

---

## Defining tools

Annotate `impl Tool for YourStruct` with `#[tool]`. Each method becomes a callable tool:

```rust
use agentix::tool;
use serde_json::{Value, json};

struct Calculator;

#[tool]
impl agentix::Tool for Calculator {
    /// Add two numbers.
    /// a: first number
    /// b: second number
    async fn add(&self, a: i64, b: i64) -> i64 {
        a + b
    }

    /// Multiply two numbers.
    /// a: first number
    /// b: second number
    async fn multiply(&self, a: i64, b: i64) -> Result<i64, String> {
        if a == 0 || b == 0 {
            Err("Multiplication by zero is boring".to_string())
        } else {
            Ok(a * b)
        }
    }
}

let agent = agentix::deepseek("sk-...")
    .tool(Calculator);
```

- Doc comment → tool description
- `/// param: description` lines → argument descriptions
- Return type just needs to implement `serde::Serialize`

---

## Memory

Two built-in memory backends, or implement [`Memory`] yourself:

```rust
use agentix::{InMemory, SlidingWindow};

// Keep all history (default)
let agent = agentix::deepseek("sk-...").memory(InMemory::new());

// Keep only the last N turns
let agent = agentix::deepseek("sk-...").memory(SlidingWindow::new(20));
```

---

## EventBus — observability

Every agent publishes all events to its [`EventBus`]. Tap any bus without affecting the agent:

```rust
// Subscribe (get a Receiver)
let mut rx = agent.subscribe();

// Tap with an async callback (spawns a background task)
agent.event_bus().tap(|msg| async move {
    if let Msg::Token(t) = msg { print!("{t}"); }
});

// Assembled stream — one Token per turn instead of many chunks
use futures::StreamExt;
let mut stream = agent.event_bus().subscribe_assembled();
while let Some(msg) = stream.next().await {
    match msg {
        Msg::Token(full) => println!("Response: {full}"),
        Msg::Done        => break,
        _                => {}
    }
}
```

---

## Graph — multi-agent pipelines

Wire [`Node`]s together with [`Graph`]. Each agent is a `Node` (has `input()` and `output()`).

`Graph::edge(&from, &to)` reads `from`'s assembled output and feeds it as a user message into `to`'s input:

```rust
use agentix::{Graph, PromptTemplate, OutputParser};

// Simple two-agent chain
let summariser  = agentix::deepseek("sk-...").system_prompt("Summarise in one sentence.");
let translator  = agentix::deepseek("sk-...").system_prompt("Translate to French.");

Graph::new()
    .edge(&summariser, &translator);

summariser.send("Long article text…").await;
// translator automatically receives the summarised text
```

### PromptTemplate

A lightweight [`Node`] that renders a template before forwarding:

```rust
let prompt = PromptTemplate::new("Translate the following to {lang}:\n{input}")
    .var("lang", "Japanese");

let agent = agentix::deepseek("sk-...");

Graph::new().edge(&prompt, &agent);

// Send a raw user message into the template
prompt.input().send(Msg::User(vec![UserContent::Text("Hello world".into())])).await.unwrap();
// agent receives: "Translate the following to Japanese:\nHello world"
```

Variables: `{input}` is replaced by the incoming `Msg::User` text; other `{key}` placeholders are pre-set with `.var(key, value)`.

### OutputParser

A lightweight [`Node`] that transforms assembled text before forwarding:

```rust
let agent  = agentix::deepseek("sk-...")
    .system_prompt("Respond with only JSON: {\"score\": <0-10>}");
let parser = OutputParser::new(|s| {
    serde_json::from_str::<serde_json::Value>(&s)
        .ok()
        .and_then(|v| v["score"].as_i64().map(|n| n.to_string()))
        .unwrap_or("0".into())
});

Graph::new().edge(&agent, &parser);
// parser.output() emits Msg::User(vec!["7".into()]) (or whatever the model returned)
```

### Middleware

Middlewares run on every message crossing any edge. Return `None` to drop:

```rust
Graph::new()
    .middleware(|msg| {
        println!("[graph] {msg:?}");
        Some(msg)
    })
    .middleware(|msg| {
        // drop empty messages
        if let Msg::User(ref parts) = msg {
            let empty = parts.iter().all(|p| matches!(p, agentix::UserContent::Text(t) if t.trim().is_empty()));
            if empty { return None; }
        }
        Some(msg)
    })
    .edge(&a, &b)
    .edge(&b, &c);
```

### Full pipeline

```rust
let prompt  = PromptTemplate::new("Score this review (0-10):\n{input}");
let scorer  = agentix::deepseek("sk-...").system_prompt("Return only JSON: {\"score\": N}");
let parser  = OutputParser::new(extract_score);
let logger  = agentix::deepseek("sk-...").system_prompt("Log: score received was {input}");

Graph::new()
    .middleware(|msg| { log::debug!("{msg:?}"); Some(msg) })
    .edge(&prompt,  &scorer)
    .edge(&scorer,  &parser)
    .edge(&parser,  &logger);

prompt.input().send(Msg::User(vec!["Great product!".into()])).await.unwrap();
```

---

## Custom Node

Implement [`Node`] to plug any async processor into a graph:

```rust
use agentix::{Node, EventBus, Msg};
use tokio::sync::mpsc;

struct UpperCaseNode { tx: mpsc::Sender<Msg>, bus: EventBus }

impl UpperCaseNode {
    fn new() -> Self {
        let (tx, mut rx) = mpsc::channel(64);
        let bus = EventBus::new(512);
        let bus_c = bus.clone();
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                let out = match msg {
                    Msg::User(parts) => Msg::User(
                        parts.into_iter()
                            .map(|p| match p {
                                agentix::UserContent::Text(t) => agentix::UserContent::Text(t.to_uppercase()),
                                other => other,
                            })
                            .collect()
                    ),
                    other        => other,
                };
                bus_c.send(out);
            }
        });
        Self { tx, bus }
    }
}

impl Node for UpperCaseNode {
    fn input(&self)  -> mpsc::Sender<Msg> { self.tx.clone() }
    fn output(&self) -> EventBus           { self.bus.clone() }
}
```

---

## Multimodal (vision)

Send images alongside text using `send_parts`:

```rust
use agentix::{ImageContent, ImageData, UserContent};

// URL image
agent.send_parts(vec![
    UserContent::Image(ImageContent {
        data: ImageData::Url("https://example.com/chart.png".into()),
        mime_type: "image/png".into(),
    }),
    UserContent::Text("Describe this chart.".into()),
]).await;

// Base64 image
let bytes = std::fs::read("photo.jpg").unwrap();
agent.send_parts(vec![
    UserContent::Image(ImageContent {
        data: ImageData::Base64(base64::encode(&bytes)),
        mime_type: "image/jpeg".into(),
    }),
    UserContent::Text("What's in this photo?".into()),
]).await;
```

For plain text, `agent.send("…")` still works unchanged.

---

## MCP tools

Use external processes as tools via the Model Context Protocol:

```toml
[dependencies]
agentix = { version = "0.3", features = ["mcp"] }
```

```rust
use agentix::McpTool;

let agent = agentix::deepseek("sk-...")
    .tool(McpTool::stdio("npx", &["-y", "@playwright/mcp"]).await?);
```

---

## Exposing tools as an MCP server

```toml
[dependencies]
agentix = { version = "0.3", features = ["mcp-server"] }
```

### Stdio (Claude Desktop / MCP Studio)

```rust
use agentix::{McpServer, ToolBundle, tool};

struct Calculator;

#[tool]
impl agentix::Tool for Calculator {
    /// Add two numbers.
    /// a: first operand   b: second operand
    async fn add(&self, a: f64, b: f64) -> f64 { a + b }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    McpServer::new(ToolBundle::new().with(Calculator))
        .with_name("my-calc")
        .serve_stdio()
        .await
}
```

### HTTP (Streamable HTTP transport)

```rust
McpServer::new(ToolBundle::new().with(MyTools))
    .serve_http("0.0.0.0:3000")
    .await?;
```

### Custom Axum routing

```rust
use agentix::{McpServer, ToolBundle};

let service = McpServer::new(ToolBundle::new().with(MyTools))
    .into_http_service(Default::default());

let router = axum::Router::new().nest_service("/mcp", service);
axum::serve(tokio::net::TcpListener::bind("0.0.0.0:3000").await?, router).await?;
```

---

## Contributing

PRs welcome. Keep changes focused; update public API docs when behaviour changes.

## License

MIT OR Apache-2.0
