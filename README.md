# agentix

[![crates.io](https://img.shields.io/crates/v/agentix.svg)](https://crates.io/crates/agentix)
[![docs.rs](https://docs.rs/agentix/badge.svg)](https://docs.rs/agentix)
[![license](https://img.shields.io/crates/l/agentix.svg)](LICENSE)

Multi-provider LLM client for Rust — streaming, non-streaming, tool calls, agentic loops, and MCP support.

DeepSeek · OpenAI · Anthropic · Gemini · Kimi · GLM · MiniMax · Grok — one unified API.

---

### Philosophy: Stream as Agent Structure

> An agent is not an object. It is a **Stream**.

agentix models agents as lazy, composable streams rather than stateful objects or DAG frameworks:

```rust
// token-level stream — full control, live progress
let mut stream = agent(tools, http, request, history, None);
while let Some(event) = stream.next().await { ... }

// turn-level stream — one CompleteResponse per LLM turn
let result = agent_turns(tools, http, request, history, None)
    .last_content().await;

// multi-agent pipeline — just Rust concurrency
let findings = join_all(questions.iter().map(|q| {
    agent_turns(tools.clone(), http.clone(), request.clone(), vec![q], None)
        .last_content()
})).await;
```

Concurrency is `join_all`. Pipelines are sequential `.await`. No orchestrator, no DAG, no magic — just streams composed with ordinary Rust.

---

### vs. other frameworks

| | agentix | rig | llm-chain | LangGraph |
|---|---|---|---|---|
| Language | Rust | Rust | Rust | Python |
| Agentic loop | ✅ `agent()` | manual | manual | ✅ graph nodes |
| Multi-agent pipeline | ✅ `join_all` + streams | manual | manual | ✅ graph edges |
| Streaming tokens | ✅ | ✅ | ❌ | ✅ |
| Streaming tool calls | ✅ | ❌ | ❌ | ❌ |
| MCP support | ✅ | ❌ | ❌ | ✅ (partial) |
| Proc-macro tools | ✅ `#[tool]` | ✅ `#[tool]` | ❌ | ❌ |
| Concurrent tool execution | ✅ | ❌ | ❌ | ✅ |
| Provider support | 8 | 10+ | 4 | 30+ |
| Agent abstraction | Stream | Object | Chain | DAG |

**vs LangGraph**: LangGraph models agents as DAGs with explicit nodes and edges. agentix models them as Streams — no graph definition, no state schema, no framework lock-in. Multi-agent pipelines are just `join_all` and sequential `.await`.

**vs rig's `#[tool]`**: Both use proc-macros, but rig requires one struct per tool. agentix lets you group multiple tools in a single `impl` block and share state (e.g. a DB connection) across them:

```rust
// rig: one #[rig_tool] per function, descriptions in attribute params,
//      return type must be Result<T, rig::tool::ToolError>
#[rig_tool(
    description = "Add two numbers",
    params(a = "first number", b = "second number")
)]
fn add(a: i32, b: i32) -> Result<i32, rig::tool::ToolError> { Ok(a + b) }

#[rig_tool(
    description = "Multiply two numbers",
    params(a = "first number", b = "second number")
)]
fn multiply(a: i32, b: i32) -> Result<i32, rig::tool::ToolError> { Ok(a * b) }

// agentix: multiple tools in one impl block, descriptions from doc comments,
//          return any type (or Result<T, String>), share state via &self
struct MathTools { precision: u8 }

#[tool]
impl Tool for MathTools {
    /// Add two numbers.
    /// a: first number  b: second number
    async fn add(&self, a: f64, b: f64) -> f64 { ... }

    /// Multiply two numbers.
    /// a: first number  b: second number
    async fn multiply(&self, a: f64, b: f64) -> f64 { ... }
}

// standalone fn — zero boilerplate, doc comment = description
/// Square root of x.
/// x: input value
#[tool]
async fn sqrt(x: f64) -> f64 { x.sqrt() }

// compose with + operator
let bundle = sqrt + MathTools { precision: 4 };
```

## Installation

```toml
[dependencies]
agentix = "0.9"

# Optional: Model Context Protocol (MCP) tool support
# agentix = { version = "0.9", features = ["mcp"] }
```

---

## Quick Start

```rust
use agentix::{Request, LlmEvent};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let http = reqwest::Client::new();

    let mut stream = Request::deepseek(std::env::var("DEEPSEEK_API_KEY")?)
        .system_prompt("You are a helpful assistant.")
        .user("What is the capital of France?")
        .stream(&http)
        .await?;

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

Eight built-in providers, all using the same API:

```rust
use agentix::Request;

// Shortcut constructors (provider + default model in one call)
let req = Request::deepseek("sk-...");
let req = Request::openai("sk-...");
let req = Request::anthropic("sk-ant-...");
let req = Request::gemini("AIza...");
let req = Request::kimi("...");       // Moonshot AI — kimi-k2.5
let req = Request::glm("...");        // Zhipu AI — glm-5
let req = Request::minimax("...");    // MiniMax — MiniMax-M2.7 (Anthropic API)
let req = Request::grok("xai-...");

// Any OpenAI-compatible endpoint (e.g. OpenRouter)
let req = Request::openai("sk-or-...")
    .base_url("https://openrouter.ai/api/v1")
    .model("openrouter/free");
```

---

## Request API

`Request` is a self-contained value type — it carries provider, credentials, model,
messages, tools, and tuning. Call `stream()` or `complete()` with a shared `reqwest::Client`.

### `stream()` — streaming completion

```rust
let http = reqwest::Client::new();
let mut stream = Request::new(Provider::OpenAI, "sk-...")
    .system_prompt("You are helpful.")
    .user("Hello!")
    .stream(&http)
    .await?;

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
let resp = Request::new(Provider::OpenAI, "sk-...")
    .user("What is 2+2?")
    .complete(&http)
    .await?;
println!("{}", resp.content.unwrap_or_default());
println!("reasoning: {:?}", resp.reasoning);
println!("tool_calls: {:?}", resp.tool_calls);
println!("usage: {:?}", resp.usage);
```

### Builder methods

```rust
let req = Request::new(Provider::DeepSeek, "sk-...")
    .model("deepseek-reasoner")
    .base_url("https://custom.api/v1")
    .system_prompt("You are helpful.")
    .max_tokens(4096)
    .temperature(0.7)
    .retries(5, 2000)           // max retries, initial delay ms
    .user("Hello!")             // convenience for adding a user message
    .message(msg)               // add any Message variant
    .messages(vec![...])        // set full history
    .tools(tool_defs);          // set tool definitions
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

Two styles are supported: **standalone function** (simpler) and **impl block** (multiple tools in one struct).

### Standalone function

```rust
use agentix::tool;

/// Add two numbers.
/// a: first number
/// b: second number
#[agentix::tool]
async fn add(a: i64, b: i64) -> i64 {
    a + b
}

/// Divide a by b.
#[agentix::tool]
async fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 { Err("division by zero".into()) } else { Ok(a / b) }
}

// Combine with + operator
let tools = add + divide;
let mut stream = agentix::agent(tools, http, request, history, Some(25_000));
```

The macro generates a unit struct with the same name as the function and implements `Tool` for it.

### Impl block (multiple methods per struct)

```rust
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

### Runtime add / remove

```rust
let mut bundle = agentix::ToolBundle::default();
bundle += Calculator;          // AddAssign — add tool in-place
bundle -= Calculator;          // SubAssign — remove all functions Calculator provides
let bundle2 = bundle + Calculator - Calculator;  // Sub — returns new bundle
```

---

## Structured Output

Constrain the model to emit JSON matching a Rust struct using `Request::json_schema()`.
Derive `schemars::JsonSchema` on your struct and pass the generated schema:

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, JsonSchema)]
struct Review {
    rating: f32,
    summary: String,
    pros: Vec<String>,
}

let schema = serde_json::to_value(schemars::schema_for!(Review))?;

let response = Request::openai(api_key)
    .system_prompt("You are a film critic.")
    .user("Review Inception (2010).")
    .json_schema("review", schema, true)   // strict=true enforces the schema
    .complete(&http)
    .await?;

let review: Review = response.json()?;
```

See `examples/08_structured_output.rs` for a runnable example.

**Provider support:**
- **OpenAI** — full `json_schema` support (gpt-4o and later)
- **Gemini** — `responseSchema` + `responseMimeType: application/json` (fully supported)
- **DeepSeek** — `json_object` only; `json_schema` is automatically degraded with a `tracing::warn`
- **Anthropic** — `response_format` is ignored; use prompt engineering instead

---

## Reliability

- **Automatic retries** — exponential backoff for 429 / 5xx responses
- **Usage tracking** — per-request token accounting across all providers; `AgentEvent::Done` contains cumulative totals across all turns

---

## Agent (agentic loop)

`agentix::agent()` drives the full LLM ↔ tool-call loop and yields typed `AgentEvent`s.
Pass it a `ToolBundle`, a base `Request`, and an initial history — it handles
repeated LLM calls, tool execution, and history accumulation automatically.

```rust
use agentix::{AgentEvent, Request, Provider, ToolBundle};
use futures::StreamExt;

#[tokio::main]
async fn main() {
    let http = reqwest::Client::new();
    let request = Request::new(Provider::DeepSeek, std::env::var("DEEPSEEK_API_KEY").unwrap())
        .system_prompt("You are helpful.");

    let mut stream = agentix::agent(ToolBundle::default(), http, request, vec![], None);
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::Token(t)                          => print!("{t}"),
            AgentEvent::ToolCallStart(tc)                 => println!("→ {}({})", tc.name, tc.arguments),
            AgentEvent::ToolResult { name, content, .. }  => println!("← [{name}] {content}"),
            AgentEvent::Usage(u)                          => println!("tokens: {}", u.total_tokens),
            AgentEvent::Error(e)                          => eprintln!("error: {e}"),
            _ => {}
        }
    }
}
```

### AgentEvent variants

- `Token(String)` — incremental response text
- `Reasoning(String)` — thinking trace
- `ToolCallChunk(ToolCallChunk)` — streaming partial tool call
- `ToolCallStart(ToolCall)` — complete tool call, about to execute
- `ToolProgress { id, name, progress }` — intermediate tool output
- `ToolResult { id, name, content }` — final tool result
- `Usage(UsageStats)` — token usage per LLM request
- `Done(UsageStats)` — emitted once when the loop finishes normally; contains **cumulative** totals across all turns
- `Warning(String)` — recoverable stream error
- `Error(String)` — fatal error

`agentix::agent()` returns a `BoxStream<'static, AgentEvent>` — drop it to abort.

---

## License

MIT OR Apache-2.0
