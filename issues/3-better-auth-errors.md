## Summary

Improve error messages for authentication failures (401/403) to make debugging easier.

## Current behavior

```
ApiError: HTTP 401
```

## Expected behavior

```
ApiError: Authentication failed (HTTP 401) — check your API key
```

## Task

In `agentix/src/provider.rs`, the `post_streaming` and `post_json` functions handle non-2xx responses. Detect 401/403 specifically and return a descriptive message.

```rust
401 => Err(ApiError::Http(401, "Authentication failed — check your API key".into())),
403 => Err(ApiError::Http(403, "Permission denied — check your quota or key permissions".into())),
```

## Acceptance criteria

- [ ] 401 responses include a message mentioning authentication / API key
- [ ] 403 responses include a message mentioning permission or quota
- [ ] `cargo check` passes
- [ ] No existing tests broken
