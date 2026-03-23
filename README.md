# agentix

[![crates.io](https://img.shields.io/crates/v/agentix.svg)](https://crates.io/crates/agentix)
[![docs.rs](https://img.shields.io/docsrs/agentix)](https://docs.rs/agentix)
[![license](https://img.shields.io/crates/l/agentix.svg)](https://github.com/ozongzi/agentix/blob/main/LICENSE-MIT)

A Rust framework for building LLM agents and multi-agent pipelines. Supports DeepSeek, OpenAI, Anthropic (Claude), and Google Gemini out of the box — plus any OpenAI-compatible endpoint.

Built on a **pure stream-based architecture**. Agents and nodes are stream transformers ([`Node`](#nodes--composition)) that can be easily chained to build complex, branching, and looping workflows using native Rust control flow.

---

## Quickstart

```bash
export DEEPSEEK_API_KEY="sk-..."
```

```toml
[dependencies]
agentix = "0.5"
tokio   = { version = "1", features = ["full"] }
futures = "0.3"
```

```rust
use agentix::AgentEvent;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = agentix::deepseek(std::env::var("DEEPSEEK_API_KEY")?)
        .system_prompt("You are a helpful assistant.");

    let mut stream = agent.chat("What is the capital of France?").await?;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::Token(t) => print!("{t}"),
            AgentEvent::Error(e) => { eprintln!("Error: {e}"); break; }
            _ => {}
        }
    }
    println!();
    Ok(())
}
```

---

## Providers

Four built-in providers, all using the same builder API:

```rust
// DeepSeek  (default model: deepseek-chat)
let mut agent = agentix::deepseek("sk-...");

// OpenAI  (default model: gpt-4o)
let mut agent = agentix::openai("sk-...");

// Anthropic / Claude  (default model: claude-opus-4-5)
let mut agent = agentix::anthropic("sk-ant-...");

// Gemini  (default model: gemini-2.0-flash)
let mut agent = agentix::gemini("AIza...");

// Any OpenAI-compatible endpoint (e.g. OpenRouter)
let mut agent = agentix::openai("sk-or-...")
    .base_url("https://openrouter.ai/api/v1")
    .model("openrouter/free");
```

---

## Agent API

`Agent` is the primary entry point. It lazily starts an internal runtime on first use.

### `chat()` — one-shot, lazy stream

Sends a message and returns a stream of events for **this turn only**. The stream ends when `Done` is emitted.

```rust
let mut stream = agent.chat("Summarise the Rust book.").await?;
while let Some(ev) = stream.next().await {
    match ev {
        AgentEvent::Token(t) => print!("{t}"),
        AgentEvent::Done     => break,
        _ => {}
    }
}
```

### `send()` + `subscribe()` — fire-and-forget / multi-consumer

`send()` dispatches a message without waiting. `subscribe()` returns a continuous `BoxStream` that receives **all** future events and never stops at `Done`. Both `&str`/`String` and raw `AgentInput` values are accepted by `send()`.

```rust
use agentix::AgentInput;

// Send a user message
agent.send("Follow-up question").await?;

// Send an abort signal
agent.send(AgentInput::Abort).await?;

// Subscribe to the raw event stream
let mut rx = agent.subscribe();
while let Some(ev) = rx.next().await {
    if matches!(ev, AgentEvent::Done) { break; }
}
```

### `sender()` — share the channel with spawned tasks

```rust
let tx = agent.sender(); // mpsc::Sender<AgentInput>
tokio::spawn(async move {
    tokio::time::sleep(Duration::from_secs(5)).await;
    tx.send(AgentInput::Abort).await.ok();
});
```

### `add_tool()` — add tools at runtime

```rust
// Before first use (builder-style)
let mut agent = agentix::deepseek("sk-...").tool(MyTool);

// After first use (async, takes effect immediately)
agent.add_tool(AnotherTool).await;
```

### `usage()` — token accounting

```rust
println!("{:?}", agent.usage()); // UsageStats { prompt_tokens, completion_tokens, ... }
```

---

## Events — The Communication Layer

### `AgentInput` (what you send)
- `User(Vec<UserContent>)` — new conversation turn (also `From<&str>` / `From<String>`)
- `ToolResult { call_id, result }` — provide a tool execution result
- `Abort` — immediately stop current processing

### `AgentEvent` (what you receive)
- `Token(String)` — incremental response text
- `Reasoning(String)` — thinking/reasoning trace (e.g. DeepSeek-R1)
- `ToolCall(ToolCall)` — model wants to call a tool
- `ToolProgress { name, progress, .. }` — streaming tool output
- `ToolResult { name, result, .. }` — tool finished
- `Usage(UsageStats)` — token usage for this turn
- `Done` — turn complete
- `Error(String)` — an error occurred

---

## Defining Tools

Annotate `impl Tool for YourStruct` with `#[tool]`. Each method becomes a callable tool.

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

    /// Divide a by b.
    async fn divide(&self, a: f64, b: f64) -> Result<f64, String> {
        if b == 0.0 { Err("division by zero".into()) } else { Ok(a / b) }
    }
}

let mut agent = agentix::deepseek("sk-...").tool(Calculator);
```

- Doc comment → tool description
- `/// param: description` lines → argument descriptions
- `Result::Err` automatically propagates as `{"error": "..."}` to the LLM

### Streaming tools

Add `#[streaming]` to a method inside `#[tool]`. No return type annotation needed — the macro infers it. Use `async_stream::stream!` in the body for `yield` syntax.

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

## Memory & Context

```rust
use agentix::{SlidingWindow, TokenSlidingWindow, LlmSummarizer};

// Keep the last N turns
let mut agent = agentix::deepseek("sk-...").memory(SlidingWindow::new(20));

// Keep up to N tokens of history
let mut agent = agentix::deepseek("sk-...").memory(TokenSlidingWindow::new(4000));

// Auto-summarise old messages with the LLM when exceeding N tokens
let mut agent = agentix::deepseek("sk-...").memory(LlmSummarizer::new(client, 8000));
```

---

## Nodes & Composition

For advanced multi-agent pipelines, use [`AgentNode`] (a raw stream transformer) and compose it with other [`Node`]s.

```rust
use agentix::{Node, AgentNode, PromptNode};

let prompt_node = PromptNode::new("Summarise in one sentence: {input}");

// Chain: String -> PromptNode -> AgentInput -> AgentNode -> AgentEvent
let input = futures::stream::iter(vec!["Long article...".to_string()]).boxed();
let agent_input = prompt_node.run(input);
let mut output = scorer_node.run(agent_input);
```

---

## Reliability

- **Automatic retries** — exponential backoff for 429 / 5xx responses
- **HTTP timeouts** — 10 s connect, 120 s response (overridable via `Agent::with_http`)
- **Concurrent tool execution** — multiple tool calls in one turn run in parallel
- **Safe memory truncation** — `SlidingWindow` never splits `tool_call` / `tool_result` pairs
- **Usage tracking** — per-turn and cumulative token accounting across all providers

---

## MCP Tools

Use external processes as tools via the Model Context Protocol:

```rust
use agentix::McpTool;
use std::time::Duration;

let tool = McpTool::stdio("npx", &["-y", "@playwright/mcp"]).await?
    .with_timeout(Duration::from_secs(60));

let mut agent = agentix::deepseek("sk-...").tool(tool);
```

---

## Changelog

### 0.5.0

- **New `Agent` API** — `chat()`, `send()`, `subscribe()`, `sender()`, `add_tool()`, `abort()`, `usage()`
  - `chat(text)` returns a lazy `BoxStream` (ends at `Done`), backed by `tokio::broadcast`
  - `send(input)` accepts `&str`, `String`, or `AgentInput` directly (`From` impls)
  - `subscribe()` returns a continuous `BoxStream` that never stops at `Done`
  - `add_tool()` inserts tools into the live registry after the runtime has started
- **Concurrent tool execution** — multiple tool calls in a single turn now run via `FuturesUnordered`
- **`SlidingWindow` fix** — truncation now skips orphaned `ToolResult` / `tool_call` messages to avoid malformed histories
- **`LlmSummarizer` fix** — summary is injected as a `user`/`assistant` pair, satisfying strict alternating-role providers (Anthropic, Gemini)
- **`estimate_tokens` fix** — BPE tokeniser is now initialised once via `OnceLock` instead of being rebuilt on every call
- **Default HTTP timeouts** — 10 s connect timeout, 120 s response timeout
- **Removed** `Session` abstraction — `Agent` manages the runtime directly

### 0.4.x

- Initial `Session`-based multi-turn API
- DeepSeek, OpenAI, Anthropic, Gemini providers
- `#[tool]` and `#[streaming_tool]` macros
- Memory backends: `InMemory`, `SlidingWindow`, `TokenSlidingWindow`, `LlmSummarizer`
- MCP tool support

---

## Contributing

PRs welcome. Built with 🦀 in Rust.

## License

MIT OR Apache-2.0
