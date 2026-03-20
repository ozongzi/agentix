use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Thinking {
    /// If set to `enabled`, the thinking (reasoning) mode will be used. If set to `disabled`, the non-thinking mode will be used.
    pub r#type: ThinkingType,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingType {
    Disabled,
    Enabled,
}
