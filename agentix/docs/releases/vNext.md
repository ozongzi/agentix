# vNext — Release notes

Version: vNext (breaking)
Date: 2026-02-28

## Summary

This release is a deliberate, breaking refactor of the crate into a clear four-layer architecture:

- raw — low-level, API-shaped request/response types (kept for advanced use).
- api — safe, chainable `ApiRequest` builder and `ApiClient`.
- conversation — `DeepseekConversation` that manages history and automatic summarization via a `Summarizer` trait.
- agent — `DeepseekAgent` built on conversation; orchestrates tool usage and multi-step agent flows.

The goal is to provide a safer, clearer, and more extensible API surface while keeping the raw types available for advanced users. The refactor also introduces a pluggable summarizer abstraction and changes the agent tool-call lifecycle (preview + execution).

> This release is breaking. Read the "Breaking changes" and "Migration" sections carefully.

---

## Breaking changes (at-a-glance)

- Removed legacy high-level types:
  - `Request` and `DeepseekClient` (old `request.rs`) — removed.
  - `NormalChatter` and `SimpleChatter` — removed.
- `Model` enum is intentionally not exported as part of the primary public API. Use:
  - `ApiRequest::deepseek_chat(...)`
  - `ApiRequest::deepseek_reasoner(...)`
- Unsafe helpers removed from the public API (e.g. `from_raw_unchecked`, `get_raw_mut`).
- Agent tool-call behavior changed:
  - When a model *requests* tools, the agent stream now yields twice:
    1. First yield: assistant content + tool-call preview events (preview result is `null`).
    2. Second yield: tool execution results (and those results are appended to the conversation history).
- `raw` module remains available as `ds_api::raw` but is no longer the recommended primary surface.

---

## What changed (details)

### New primary API

- `ApiRequest` (in `src/api.rs`)
  - Chainable builder pattern for safe request construction.
  - Two model-select helpers: `deepseek_chat(...)` and `deepseek_reasoner(...)`.
  - Methods: `.messages(...)`, `.add_message(...)`, `.json()`, `.text()`, `.temperature()`, `.max_tokens()`, `.add_tool(...)`, `.tool_choice_auto()`, `.stream(bool)`, etc.

- `ApiClient` (in `src/api.rs`)
  - Lightweight client owning token/base_url/reqwest client.
  - Methods:
    - `send(ApiRequest) -> ChatCompletionResponse` (non-streaming)
    - `send_stream(ApiRequest) -> Stream<ChatCompletionChunk>` (SSE streaming)
    - `stream_text(ApiRequest) -> Stream<Result<String, ApiError>>` (text fragments from streaming)

### Conversation and Summarizer

- `Summarizer` trait (in `src/conversation/mod.rs`)
  - Pluggable abstraction for summarizing conversation history.
  - Default provided: `TokenBasedSummarizer`.

- `TokenBasedSummarizer` (default)
  - Estimates tokens roughly as `chars / 4`.
  - SKIPS `system` messages when estimating (system prompts are not counted).
  - Default threshold: 100,000 estimated tokens (configurable).
  - When triggered, older messages are compressed into a single `system` message (marked via `name` as `[auto-summary]`).

- `DeepseekConversation`
  - Manages history and auto-summary.
  - Methods: `push_user_input(String)`, `add_message(Message)`, `send_once()`, `stream_text()` (inherent async), builder helpers like `.with_summarizer(...)`, `.enable_auto_summary(...)`.

### Agent behavior

- `DeepseekAgent`
  - Wraps a `DeepseekConversation`, `ApiClient`, and zero or more tools (`Tool` trait).
  - `add_tool(...)` registers tool functions (via `#[tool]` macro).
  - `with_system_prompt(...)` lets you set a system prompt before starting the conversation.
  - `chat(...)` returns an `AgentStream` (implements `Stream<Item = Result<AgentResponse, ApiError>>`).
  - `.with_streaming()` — enables SSE-based streaming; text fragments are yielded as they arrive.

- `AgentStream` lifecycle (important)
  - `AgentStream` now yields `Result<AgentResponse, ApiError>` instead of `AgentResponse`. API and network errors are propagated as `Err(e)` items rather than silently terminating the stream.
  - When the assistant response contains tool-call requests, the agent yields:
    1. Assistant content + preview of tool calls (preview result is `null`) — so callers can display the assistant reply and the fact that tools will be invoked.
    2. After the agent runs the tools, the agent yields tool-call events with actual results; these results are appended to the conversation history as `Role::Tool` messages.
  - `.with_streaming()` — opt-in streaming mode. When enabled the agent uses SSE internally and yields text fragments one by one as they arrive instead of waiting for the full response. Tool call handling is otherwise identical.

- Tool argument error handling
  - The `#[tool]` macro now generates `match`-based argument parsing. If the model provides an argument with the wrong type or missing required fields, the tool returns `{"error": "invalid argument '...': ..."}` to the conversation history instead of panicking. The model can then see the error and retry with corrected arguments.

- `#[tool]` macro JSON schema generation
  - Replaced string-based type matching with recursive `syn::Type` structural matching.
  - `Vec<T>` now correctly generates `{"type": "array", "items": <T schema>}`.
  - Path-qualified types such as `std::string::String` now resolve to `"string"`.
  - All integer primitives (`u8`–`u128`, `i8`–`i128`, `usize`, `isize`) are explicitly matched instead of relying on fallthrough.

---

## Migration guide

### If you used `Request`/`DeepseekClient`

Old:
```rust
use ds_api::request::Request;
use ds_api::request::DeepseekClient;

let req = Request::basic_query(vec![ Message::new(Role::User, "Hello") ]);
let client = DeepseekClient::new(token);
let resp = client.send(req).await?;
```

New:
```rust
use ds_api::{ApiClient, ApiRequest};
use ds_api::raw::deepseek::request::message::Message;

let client = ApiClient::new(token);
let req = ApiRequest::deepseek_chat(vec![ Message::new(ds_api::raw::deepseek::request::message::Role::User, "Hello") ]);
let resp = client.send(req).await?;
```

### If you used `NormalChatter` / `SimpleChatter`

Old:
```rust
use ds_api::NormalChatter;
let mut chatter = NormalChatter::new(token);
let mut history = vec![/*...*/];
let resp = chatter.chat("Hello", &mut history).await?;
```

New:
```rust
use ds_api::{ApiClient, DeepseekConversation, Message, raw::deepseek::request::message::Role};

let client = ApiClient::new(token);
let mut conv = DeepseekConversation::new(client.clone())
    .with_history(vec![Message::new(Role::System, "You are helpful.")]);

conv.push_user_input("Hello".to_string());
let reply = conv.send_once().await?;
```

### Agent & tools (migration)

- Tools still use the `#[tool]` macro and `Tool` trait.
- Build agent, add tools, optionally set system prompt, then `chat(...)` to get a streaming `AgentResponse` sequence.

Example:
```rust
let agent = DeepseekAgent::new(token)
    .add_tool(MyTool { /* ... */ })
    .with_system_prompt("You are a helpful assistant.");
let mut stream = agent.chat("What's the weather in Tokyo?");
while let Some(event) = stream.next().await {
    match event {
        Err(e) => { eprintln!("Error: {e}"); break; }
        Ok(ev) => {
            if let Some(content) = &ev.content { println!("assistant: {}", content); }
            for tc in &ev.tool_calls { println!("tool: {} -> {}", tc.name, tc.result); }
        }
    }
}
```

For streaming text fragments as they arrive:
```rust
let agent = DeepseekAgent::new(token).with_streaming().add_tool(MyTool { /* ... */ });
let mut stream = agent.chat("What's the weather in Tokyo?");
while let Some(event) = stream.next().await {
    match event {
        Err(e) => { eprintln!("Error: {e}"); break; }
        Ok(ev) => { if let Some(fragment) = &ev.content { print!("{}", fragment); } }
    }
}
```

Important: when the model requests tool calls, the first yielded event contains the assistant's content and a preview of the tool calls. The second yielded event contains the results of tool execution.

---

## Examples and docs

- Example agent (tool demo): `ds-api/examples/agent_demo.rs` — shows `#[tool]` usage, `DeepseekAgent` creation, and streaming consumption.
- README updated with quick examples and migration notes.
- Consider reading `src/conversation/mod.rs` for details about the `Summarizer` trait and default behavior.

---

## How to test locally

- Build:
```bash
cargo build
```

- Run unit tests:
```bash
cargo test
```

- Lint (Clippy is enforced in CI):
```bash
cargo clippy -p ds-api -- -D warnings
```

- Try the example (requires a valid token and network access):
```bash
cd ds-api
cargo run --example agent_demo
```

---

## Changelog notes

- vNext (this release):
  - Full refactor to layered design.
  - Removed legacy high-level types (breaking).
  - Introduced `ApiRequest`/`ApiClient`, `DeepseekConversation`, `DeepseekAgent`.
  - Added `Summarizer` trait and `TokenBasedSummarizer` default (skips system prompts).
  - Changed agent's tool-call flow (preview + execution yields).
  - **Breaking:** `AgentStream::Item` changed from `AgentResponse` to `Result<AgentResponse, ApiError>`.
  - Added `DeepseekAgent::with_streaming()` for SSE-based text streaming.
  - `#[tool]` macro: argument parse failures now return an error JSON to the LLM instead of panicking.
  - `#[tool]` macro: JSON schema generation now uses `syn::Type` structural matching (`Vec<T>`, path-qualified types, all integer primitives handled correctly).

---

## Future work (non-blocking)

- Add an LLM-backed semantic summarizer as a built-in summarizer option.
- Provide optional thin compatibility adapters for projects that need an easier migration path.
- Add more examples and an `UPGRADING.md` with automated code transforms for common patterns.

---

## Contact / support

If you need assistance migrating or want a compatibility wrapper implemented, open an issue or request a PR. I can produce an `UPGRADING.md` and convert common usage patterns automatically if desired.
