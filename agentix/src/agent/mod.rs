/*!
Agent module

This module is split into focused submodules:

- `agent_core` — the public agent struct, event/response types and tool
  registration logic.
- `executor` — pure business-logic functions: building requests, fetching
  responses, opening SSE streams, executing tools.  No `Poll` or `Context`
  here — just `async fn`s that do real work.
- `stream` — the asynchronous `AgentStream` state machine that schedules
  calls into `executor` and drives the full agent loop.

Public types are re-exported at the crate level so callers never need to
reach into the submodules directly.
*/

pub mod agent_core;
pub(crate) mod executor;
pub mod stream;

pub use agent_core::{AgentEvent, AnthropicAgent, DeepSeekAgent, GeminiAgent, OpenAIAgent, ToolCallChunk, ToolCallResult, ToolCommand};
pub use stream::AgentStream;
