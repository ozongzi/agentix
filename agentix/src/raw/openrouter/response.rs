use serde::Deserialize;
use serde_json::Value;

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
    #[serde(alias = "reasoning_content")]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
    /// Typed reasoning entries (`reasoning.text` / `reasoning.summary` /
    /// `reasoning.encrypted`). Streamed fragmented across chunks — we
    /// accumulate by the `index` field on each entry, not by append order
    /// (see LangChain #36400 for the bug that motivates this).
    #[serde(default)]
    pub reasoning_details: Option<Vec<Value>>,
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
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

impl From<Usage> for crate::types::UsageStats {
    fn from(u: Usage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens as usize,
            completion_tokens: u.completion_tokens as usize,
            total_tokens: u.total_tokens as usize,
            cache_read_tokens: u
                .prompt_tokens_details
                .map(|d| d.cached_tokens as usize)
                .unwrap_or(0),
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
    #[serde(alias = "reasoning_content")]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<CompleteToolCall>>,
    /// Typed reasoning entries preserved for round-trip.
    #[serde(default)]
    pub reasoning_details: Option<Vec<Value>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompleteToolCall {
    pub id: String,
    pub function: CompleteFunctionCall,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompleteFunctionCall {
    pub name: String,
    pub arguments: String,
}
