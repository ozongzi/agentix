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
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// Current Kimi docs put `cached_tokens` at the TOP LEVEL of `usage`
    /// ("Number of tokens served from cache"), not under
    /// `prompt_tokens_details`.
    #[serde(default)]
    pub cached_tokens: u32,
    /// OpenAI-style nested location, kept for compatibility with older
    /// responses / proxies.
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

// Kimi billing semantics: cached tokens are a subset of `prompt_tokens`
// (cost = cached × hit-price + (prompt − cached) × input-price + output).
// Caching is automatic with no write or storage fee (the old explicit
// context-caching API with creation + per-minute storage charges is gone
// from current docs). K3 reasoning is billed inside output.
impl From<Usage> for crate::types::Usage {
    fn from(u: Usage) -> Self {
        let cached = if u.cached_tokens > 0 {
            u.cached_tokens as u64
        } else {
            u.prompt_tokens_details
                .map(|d| d.cached_tokens as u64)
                .unwrap_or(0)
        };
        Self {
            input: (u.prompt_tokens as u64).saturating_sub(cached),
            cache_read: cached,
            output: u.completion_tokens as u64,
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
