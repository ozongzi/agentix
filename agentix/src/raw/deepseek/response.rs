use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct StreamChunk {
    pub choices: Vec<ChunkChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkChoice {
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Delta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct DeltaToolCall {
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
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
            prompt_tokens: u.prompt_tokens as usize,
            completion_tokens: u.completion_tokens as usize,
            total_tokens: u.total_tokens as usize,
            cache_read_tokens: u.prompt_cache_hit_tokens as usize,
            ..Default::default()
        }
    }
}

// ── Non-streaming response ────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CompleteResponse {
    pub choices: Vec<CompleteChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct CompleteChoice {
    pub message: CompleteMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CompleteMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<CompleteToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct CompleteToolCall {
    pub id: String,
    pub function: CompleteFunctionCall,
}

#[derive(Debug, Deserialize)]
pub struct CompleteFunctionCall {
    pub name: String,
    pub arguments: String,
}
