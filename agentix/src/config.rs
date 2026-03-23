use serde_json::{Map, Value};

/// Runtime-mutable configuration for an agent turn.
///
/// Stored inside [`LlmClient`][crate::LlmClient] behind an `RwLock`.
/// Every field takes effect on the **next** API request — changes made while
/// a generation is in-flight apply from the following turn onwards.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub base_url:      String,
    pub model:         String,
    pub system_prompt: Option<String>,
    pub max_tokens:    Option<u32>,
    pub temperature:   Option<f32>,
    pub extra_body:    Map<String, Value>,

    /// Maximum number of retries for transient errors. Default: 3.
    pub max_retries:   u32,
    /// Initial delay between retries in milliseconds. Default: 1000ms.
    pub retry_delay_ms: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            base_url:      String::new(),
            model:         String::new(),
            system_prompt: None,
            max_tokens:    None,
            temperature:   None,
            extra_body:    Map::new(),
            max_retries:   3,
            retry_delay_ms: 1000,
        }
    }
}
