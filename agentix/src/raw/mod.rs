//! Raw, provider-specific request/response types.
//!
//! Each sub-module maps directly to a provider's JSON schema and is used
//! internally by the corresponding [`Provider`](crate::Provider) implementation.
//! The [`shared`] module contains types common across providers (e.g.
//! [`ToolDefinition`](shared::ToolDefinition), [`ToolChoice`](shared::ToolChoice)).
//!
//! Most users should interact through [`LlmClient`](crate::LlmClient) and never
//! need to touch these types directly.
pub mod anthropic;
pub mod deepseek;
pub mod gemini;
pub mod openai;
pub mod shared;

pub use deepseek::*;
pub use shared::{FunctionDefinition, FunctionName, ResponseFormat, ResponseFormatKind, ToolChoice, ToolChoiceFunction, ToolChoiceMode, ToolDefinition, ToolKind};
