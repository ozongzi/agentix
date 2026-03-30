## Summary

Add a `Request::openrouter()` shortcut constructor for OpenRouter, similar to existing provider shortcuts.

## Background

OpenRouter is a popular API gateway that exposes OpenAI-compatible endpoints. Currently users have to manually set the base URL:

```rust
let req = Request::openai("sk-or-...")
    .base_url("https://openrouter.ai/api/v1")
    .model("openrouter/auto");
```

## Task

1. Add `Provider::OpenRouter` variant in `agentix/src/request.rs`
2. Set `default_base_url` to `"https://openrouter.ai/api/v1"`
3. Set `default_model` to `"openai/gpt-4o"`
4. Route through `stream_openai_compatible` / `complete_openai_compatible` (same as `Provider::OpenAI`)
5. Add `Request::openrouter(api_key)` shortcut method
6. Update the Providers section in `README.md`

## Pattern to follow

Look at how `Provider::Kimi` or `Provider::Grok` were added in `agentix/src/request.rs` — this is the exact same pattern.

## Acceptance criteria

- [ ] `cargo check` passes with no errors
- [ ] `Request::openrouter("sk-or-...")` compiles and dispatches correctly
- [ ] README updated
