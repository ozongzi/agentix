use serde::Deserialize;

use crate::raw::deepseek::request::message::{Role, ToolType};

// Delta is similar to Message, but `tool_calls` are incremental (include an index)
#[derive(Debug, Deserialize)]
pub struct Delta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub role: Option<Role>,
    #[serde(default)]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct DeltaToolCall {
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub r#type: Option<ToolType>,
    #[serde(default)]
    pub function: Option<DeltaFunctionCall>,
}

#[derive(Debug, Deserialize)]
pub struct DeltaFunctionCall {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}
