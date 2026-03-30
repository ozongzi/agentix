## Summary

Add a `Request::with_header(key, value)` builder method to pass arbitrary HTTP headers to the provider.

## Motivation

Some providers and gateways require custom headers:
- OpenRouter: `HTTP-Referer` and `X-Title` for app attribution
- Self-hosted endpoints may require custom auth headers
- Enterprise proxies may need additional headers

```rust
let req = Request::openrouter("sk-or-...")
    .with_header("HTTP-Referer", "https://myapp.com")
    .with_header("X-Title", "My App");
```

## Files to touch

- `agentix/src/request.rs` — add `extra_headers: Vec<(String, String)>` field + `with_header()` builder method
- `agentix/src/provider.rs` — pass extra headers through `PostConfig`
- `agentix/src/raw/openai/mod.rs` — apply headers in `stream_openai_compatible` / `complete_openai_compatible`

## Notes

`PostConfig` already has an `extra_headers: &[(&str, &str)]` field — the main work is threading the runtime headers from `Request` through to it.

## Acceptance criteria

- [ ] `cargo check` passes
- [ ] Custom headers are sent in the HTTP request
- [ ] Existing tests still pass (`cargo test`)
