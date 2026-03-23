# v0.4.0 — Release notes

Version: 0.4.0 (breaking)
Date: 2026-03-23

## Summary

This release marks the evolution of the crate into a multi-provider LLM agent framework. We have unified the API across DeepSeek, OpenAI, Anthropic, and Gemini, and introduced significant improvements to the tool-calling system.

The core goal of this release is to provide a robust, type-safe, and idiomatic Rust experience for building LLM agents.

---

## Key Features & Improvements

### 1. Robust Tool Macro (`#[tool]`)
- **dtolnay trick implementation**: The `#[tool]` macro now intelligently handles both plain return types (`T`) and `Result<T, E>`.
- **Automatic Error Propagation**: When a tool returns `Result::Err(e)`, it is automatically converted to `{"error": e.to_string()}` and sent back to the LLM. This encourages models to self-correct based on readable error messages.
- **Type Safety**: Input parameters are statically checked for `JsonSchema` and `Deserialize` implementation.
- **Recursive Schema Generation**: Improved JSON Schema generation for complex types like `Vec<T>`, `Option<T>`, and nested structs.

### 2. Multi-Provider Support
- Unified `LlmClient` and `Agent` abstractions for:
  - **DeepSeek** (chat and reasoner)
  - **OpenAI** (GPT-4o, etc.)
  - **Anthropic** (Claude 3.5 Sonnet/Opus)
  - **Gemini** (2.0 Flash/Pro)

### 3. Shared Context & Memory
- Introduced `SharedContext` for thread-safe state management across agents.
- Pluggable `Memory` traits (InMemory, SlidingWindow) for managing conversation history.

### 4. Re-exported Dependencies
- `agentix` now re-exports `serde`, `serde_json`, `async_trait`, and `schemars`. Users no longer need to manually add these to their `Cargo.toml` to use the `#[tool]` macro.

---

## Breaking Changes

- **Macro Expansion Path**: The `#[tool]` macro now generates code referencing `agentix::...`. You must ensure `agentix` is available in your scope.
- **Result Serialization**: Returning `Result<T, E>` from a tool no longer serializes as `{"Ok": ...}` or `{"Err": ...}`. It now transparently returns the success value or an `{"error": "..."}` object.
- **Error Constraint**: Any error type `E` returned in a `Result<T, E>` from a tool must now implement `std::fmt::Display`.

---

## Quick Migration

### Old Tool (v0.3.x)
```rust
#[tool]
impl Tool for MyTool {
    async fn run(&self, input: String) -> serde_json::Value {
        serde_json::json!({ "output": input })
    }
}
```

### New Tool (v0.4.0)
```rust
#[tool]
impl Tool for MyTool {
    async fn run(&self, input: String) -> String {
        input // Plain types just work
    }
}
// OR with error handling
#[tool]
impl Tool for MyTool {
    async fn run(&self, input: String) -> Result<String, MyError> {
        Ok(input)
    }
}
```

---

## How to Test

```bash
cargo test --workspace
```

---

## Roadmap

- MCP (Model Context Protocol) server support enhancement.
- Advanced graph-based multi-agent orchestration.
- Semantic summarizers for long-term memory.
