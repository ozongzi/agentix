# agentix

[![crates.io](https://img.shields.io/crates/v/agentix.svg)](https://crates.io/crates/agentix)
[![docs.rs](https://docs.rs/agentix/badge.svg)](https://docs.rs/agentix)
[![license](https://img.shields.io/crates/l/agentix.svg)](LICENSE)

Multi-provider LLM client for Rust — streaming, non-streaming, tool calls, agentic loops, and MCP support.

DeepSeek · OpenAI · Anthropic · Gemini · Kimi · GLM · MiniMax · Grok · OpenRouter — one unified API.

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

## vs. other frameworks

| | agentix | rig | llm-chain | LangGraph |
|---|---|---|---|---|
| Language | Rust | Rust | Rust | Python |
| Agentic loop | ✅ `agent()` | manual | manual | ✅ graph nodes |
| Multi-agent pipeline | ✅ `join_all` + streams | manual | manual | ✅ graph edges |
| Streaming tokens | ✅ | ✅ | ❌ | ✅ |
| Streaming tool calls | ✅ | ❌ | ❌ | ❌ |
| MCP support | ✅ | ❌ | ❌ | ✅ (partial) |
| Proc-macro tools | ✅ `#[tool]` | ✅ `#[rig_tool]` | ❌ | ❌ |
| Concurrent tool execution | ✅ | ❌ | ❌ | ✅ |
| Provider support | 8 | 10+ | 4 | 30+ |
| Agent abstraction | Stream | Object | Chain | DAG |

**vs LangGraph**: LangGraph models agents as DAGs with explicit nodes and edges. agentix models them as Streams — no graph definition, no state schema, no framework lock-in. Multi-agent pipelines are just `join_all` and sequential `.await`.

**vs rig's `#[rig_tool]`**: rig requires one annotated function per tool, with descriptions passed as attribute arguments and return type fixed to `Result<T, ToolError>`. agentix uses doc comments for descriptions, accepts any return type, and lets you group related tools in a single `impl` block with shared state:

```rust
// rig: one #[rig_tool] per function
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

// agentix: one #[tool] for the whole impl block, descriptions from doc comments
struct MathTools { precision: u8 }  // shared state across all methods

#[tool]
impl Tool for MathTools {
    /// Add two numbers.
    /// a: first number  b: second number
    async fn add(&self, a: f64, b: f64) -> f64 { ... }

    /// Multiply two numbers.
    /// a: first number  b: second number
    async fn multiply(&self, a: f64, b: f64) -> f64 { ... }
}

// standalone fn also works — doc comment = description
/// Square root of x.
/// x: input value
#[tool]
async fn sqrt(x: f64) -> f64 { x.sqrt() }

let bundle = sqrt + MathTools { precision: 4 };  // compose with +
```

---

## Installation

```toml
[dependencies]
agentix = "0.18.2"

# Optional: Model Context Protocol (MCP) tool support
# agentix = { version = "0.18.2", features = ["mcp"] }

# Optional: drive `claude -p` as the agentic loop using a Claude Max OAuth session
# agentix = { version = "0.18.2", features = ["claude-code"] }
```

---

## Logging Full Request / Response Bodies

Full request bodies, response bodies, streaming chunks, and MCP raw request bodies are treated as sensitive and are disabled by default.

To enable them, you must opt in at both compile time and runtime:

```bash
AGENTIX_LOG_BODIES=1 cargo run --features sensitive-logs
```

If either one is missing, agentix will not print full bodies.

- Compile-time gate: `sensitive-logs`
- Runtime gate: `AGENTIX_LOG_BODIES=1`

This affects:

- outbound HTTP request bodies
- non-streaming HTTP response bodies
- raw SSE streaming chunks
- MCP raw HTTP request bodies

---

## Providers

Nine built-in providers, all using the same API:

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
let req = Request::openrouter("sk-or-..."); // OpenRouter with prompt caching support

// Custom base URL for OpenAI-compatible endpoints
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
    .model("deepseek-v4-pro")
    .base_url("https://custom.api/v1")
    .system_prompt("You are helpful.")
    .max_tokens(4096)
    .temperature(0.7)
    .reasoning_effort(ReasoningEffort::High)
    .retries(5, 2000)           // max retries, initial delay ms
    .user("Hello!")             // convenience for adding a user message
    .message(msg)               // add any Message variant
    .messages(vec![...])        // set full history
    .tools(tool_defs);          // set tool definitions
```

---

## Reasoning control (`ReasoningEffort`)

A single cross-provider dial for "how much should the model think". Providers that expose a thinking toggle and/or effort level map this to their own wire format; providers that don't, ignore it.

```rust
use agentix::{Request, ReasoningEffort};

let req = Request::deepseek(key)
    .reasoning_effort(ReasoningEffort::Max)    // maximum effort
    .user("Prove that there are infinitely many primes.");
```

| Variant   | DeepSeek                                          | Anthropic (Claude 4.6+)           | OpenAI / Grok / others    |
|-----------|---------------------------------------------------|-----------------------------------|---------------------------|
| `None`    | `thinking.type: disabled` (sampling params valid) | `thinking.type: disabled`         | ignored                   |
| `Minimal` | `thinking.type: enabled`, effort `high`           | `adaptive`, effort `low`          | ignored                   |
| `Low`     | `thinking.type: enabled`, effort `high`           | `adaptive`, effort `low`          | ignored                   |
| `Medium`  | `thinking.type: enabled`, effort `high`           | `adaptive`, effort `medium`       | ignored                   |
| `High`    | `thinking.type: enabled`, effort `high`           | `adaptive`, effort `high`         | ignored                   |
| `XHigh`   | `thinking.type: enabled`, effort `max`            | `adaptive`, effort `xhigh`        | ignored                   |
| `Max`     | `thinking.type: enabled`, effort `max`            | `adaptive`, effort `max`          | ignored                   |
| unset     | no `thinking` field (provider default)            | no `thinking` field (off default) | no field                  |

Notes:
- **`None` vs unset matter.** `None` emits an explicit `disabled` toggle (and lets sampling params like `temperature` flow through on DeepSeek). Leaving it unset means "don't touch the field" and accepts the provider's own default — which for DeepSeek is **thinking on** and for Anthropic is **thinking off**.
- **DeepSeek forbids sampling params in thinking mode**; setting `.temperature()` while thinking is on drops temperature before the wire with a `tracing::warn!`. Use `.reasoning_effort(ReasoningEffort::None)` to re-enable sampling.
- **Anthropic round-trip for thinking + tool use** is automatic: thinking blocks (with signatures) are captured in `Message::Assistant.provider_data` and re-emitted verbatim on the next turn, preserving the interleaved `[thinking, tool_use, thinking, tool_use]` ordering that the API verifies.

See `examples/11_reasoning.rs` for a live comparison of the four states.

---

## LlmEvent (what you receive from `stream()`)

`LlmEvent` is `#[non_exhaustive]`; always include a wildcard `_ => {}` arm to stay forward-compatible.

- `Token(String)` — incremental response text
- `Reasoning(String)` — thinking/reasoning trace (e.g. DeepSeek, Claude extended thinking)
- `ToolCallChunk(ToolCallChunk)` — partial tool call for real-time UI
- `ToolCall(ToolCall)` — completed tool call
- `AssistantState(serde_json::Value)` — opaque per-turn provider state (e.g. Anthropic thinking blocks with signatures). The agent loop attaches it to `Message::Assistant.provider_data` for round-trip; most user code can ignore it.
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

## Claude Code (Max OAuth)

`Provider::ClaudeCode` is a regular provider backed by `claude -p`, so you can
ride an existing **Claude Max** subscription instead of paying per-token via
`ANTHROPIC_API_KEY`. It plugs into `agent()` like any other provider — agentix
owns the loop, tool calls dispatch locally through the `Tool` trait, and the
loopback MCP server only surfaces tool schemas. Auth comes from the CLI's
OAuth session in the OS keychain.

Requires the `claude-code` feature and the [`claude` CLI] installed + logged in.

```toml
agentix = { version = "0.18.2", features = ["claude-code"] }
```

```rust
use agentix::{AgentEvent, Message, Request, UserContent, agent, tool};
use futures::StreamExt;

struct Calculator;
#[tool]
impl agentix::Tool for Calculator {
    /// Add two numbers.  a: first  b: second
    async fn add(&self, a: f64, b: f64) -> f64 { a + b }
}

#[tokio::main]
async fn main() {
    let http = reqwest::Client::new();
    let base = Request::claude_code()
        .model("sonnet")
        .system_prompt("You are a concise math assistant. Always use tools for arithmetic.");
    let history = vec![Message::User(vec![UserContent::Text {
        text: "What is 123 + 456?".into(),
    }])];

    let mut stream = agent(Calculator, http, base, history, None);

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::Token(t) => print!("{t}"),
            AgentEvent::ToolCallStart(tc) => println!("\n→ {}({})", tc.name, tc.arguments),
            AgentEvent::Done(u) => println!("\n[tokens: {}]", u.total_tokens),
            _ => {}
        }
    }
}
```

Each turn spawns a fresh `claude -p`, replays prior history via `--resume`,
and kills the subprocess once the first assistant turn lands — so the agent
loop keeps full control over tool dispatch and multi-turn state.

See `examples/10_claude_code.rs` for a runnable example.

[`claude` CLI]: https://docs.claude.com/en/docs/claude-code/overview

---

## License

MIT OR Apache-2.0
