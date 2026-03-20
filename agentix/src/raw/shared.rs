//! Shared wire-format types used across all provider modules.
//!
//! These types are **not** tied to any specific provider.  They live here so
//! that `request.rs` (the provider-agnostic request layer) and each provider's
//! `raw/` module can all reference them without creating circular dependencies
//! or leaking DeepSeek-specific internals.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Tool definition ───────────────────────────────────────────────────────────

/// A tool (function) that the model may call.
///
/// This is the canonical representation used throughout the crate.  Each
/// provider's `raw/` module converts this into its own wire format inside
/// `From<AgentRequest>`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolDefinition {
    /// Always `"function"` — the only tool type currently supported.
    #[serde(rename = "type")]
    pub kind: ToolKind,
    pub function: FunctionDefinition,
}

impl ToolDefinition {
    /// Convenience constructor.
    pub fn function(function: FunctionDefinition) -> Self {
        Self { kind: ToolKind::Function, function }
    }
}

/// The kind of a tool.  Only `function` is currently defined.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ToolKind {
    Function,
}

/// The function definition inside a [`ToolDefinition`].
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionDefinition {
    /// Name of the function. `[a-zA-Z0-9_-]`, max 64 chars.
    pub name: String,

    /// Human-readable description used by the model to decide when to call it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema describing the function's parameters.
    pub parameters: Value,

    /// Beta: when `true` the API validates outputs against the schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

// ── Tool choice ───────────────────────────────────────────────────────────────

/// OpenAI-style tool_choice field (serialised as either a string or an object).
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum ToolChoice {
    /// One of `"none"`, `"auto"`, `"required"`.
    String(ToolChoiceMode),
    /// Force a specific function by name.
    Object(ToolChoiceFunction),
}

/// String variants of tool_choice.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    None,
    Auto,
    Required,
}

/// Object form: `{ "type": "function", "function": { "name": "..." } }`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolChoiceFunction {
    #[serde(rename = "type")]
    pub kind: ToolKind,
    pub function: FunctionName,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionName {
    pub name: String,
}

// ── Response format ───────────────────────────────────────────────────────────

/// OpenAI-style `response_format` field.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub kind: ResponseFormatKind,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormatKind {
    Text,
    JsonObject,
}

// ── Conversions from provider-agnostic types ──────────────────────────────────

impl From<crate::request::ToolChoice> for ToolChoice {
    fn from(tc: crate::request::ToolChoice) -> Self {
        match tc {
            crate::request::ToolChoice::Auto     => ToolChoice::String(ToolChoiceMode::Auto),
            crate::request::ToolChoice::None     => ToolChoice::String(ToolChoiceMode::None),
            crate::request::ToolChoice::Required => ToolChoice::String(ToolChoiceMode::Required),
            crate::request::ToolChoice::Tool(name) => ToolChoice::Object(ToolChoiceFunction {
                kind: ToolKind::Function,
                function: FunctionName { name },
            }),
        }
    }
}

impl From<crate::request::ResponseFormat> for ResponseFormat {
    fn from(f: crate::request::ResponseFormat) -> Self {
        ResponseFormat {
            kind: match f {
                crate::request::ResponseFormat::Text       => ResponseFormatKind::Text,
                crate::request::ResponseFormat::JsonObject => ResponseFormatKind::JsonObject,
            },
        }
    }
}
