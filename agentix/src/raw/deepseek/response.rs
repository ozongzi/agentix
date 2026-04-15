use serde::Deserialize;

use crate::raw::openai::response::ChunkChoice;

/// DeepSeek streaming chunk — identical to OpenAI except for the Usage shape.
#[derive(Debug, Deserialize)]
pub struct StreamChunk {
    pub choices: Vec<ChunkChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(default)]
    pub prompt_cache_hit_tokens: u32,
    #[serde(default)]
    pub prompt_cache_miss_tokens: u32,
}

impl From<Usage> for crate::types::UsageStats {
    fn from(u: Usage) -> Self {
        Self {
            prompt_tokens:     u.prompt_tokens as usize,
            completion_tokens: u.completion_tokens as usize,
            total_tokens:      u.total_tokens as usize,
            cache_read_tokens: u.prompt_cache_hit_tokens as usize,
            ..Default::default()
        }
    }
}

/// DeepSeek non-streaming response — same as OpenAI but with DeepSeek Usage.
#[derive(Debug, Deserialize)]
pub struct CompleteResponse {
    pub choices: Vec<crate::raw::openai::response::CompleteChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}
