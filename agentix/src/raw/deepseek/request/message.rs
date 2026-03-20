use serde::{Deserialize, Serialize};

pub use crate::raw::shared::ToolKind as ToolType;

// Unified message struct
// This struct is used both for the `messages` array in requests and the `message` field in responses.
// All fields are optional to cover different roles and scenarios.
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct Message {
    /// default role is User
    pub role: Role,

    /// The content may be null for assistant messages (only when `tool_calls` are present)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Optional name to identify a user or function
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Required when role = "tool"; links to the previous tool call ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Present when role = "assistant" and the model requested tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Reasoning content produced by the model (may appear only in responses)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,

    /// Beta: if true, forces the model to begin its reply with the prefix content provided in this assistant message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix: Option<bool>,
}

impl Message {
    pub fn new(role: Role, message: &str) -> Self {
        Self {
            role,
            content: Some(message.to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
            prefix: None,
        }
    }

    pub fn user(message: &str) -> Self {
        Self::new(Role::User, message)
    }

    pub fn assistant(message: &str) -> Self {
        Self::new(Role::Assistant, message)
    }

    pub fn system(message: &str) -> Self {
        Self::new(Role::System, message)
    }

}

// Role enum (includes Tool variant)
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    #[default]
    User,
    Assistant,
    Tool,
}

// Tool call struct (reused in requests and responses)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    pub r#type: ToolType,
    pub function: FunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String, // JSON string
}
