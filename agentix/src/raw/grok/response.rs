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
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Deserialize, Default)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: u32,
}

// xAI billing semantics: `cached_tokens` is a subset of `prompt_tokens`
// (automatic caching, no write fee documented). Reasoning is billed at the
// output rate, but whether `completion_tokens` numerically INCLUDES
// `reasoning_tokens` has changed across model generations (historically it
// did not — reasoning was additive). Disambiguate per response via the
// identity `total ≈ prompt + completion + reasoning`: if it holds, reasoning
// is additive and `completion_tokens` is already reasoning-free; otherwise
// reasoning is a subset and gets subtracted out.
impl From<Usage> for crate::types::Usage {
    fn from(u: Usage) -> Self {
        let cached = u
            .prompt_tokens_details
            .map(|d| d.cached_tokens as u64)
            .unwrap_or(0);
        let reasoning = u
            .completion_tokens_details
            .map(|d| d.reasoning_tokens as u64)
            .unwrap_or(0);
        let (prompt, completion, total) = (
            u.prompt_tokens as u64,
            u.completion_tokens as u64,
            u.total_tokens as u64,
        );
        let reasoning_is_additive = reasoning > 0 && total >= prompt + completion + reasoning;
        let output = if reasoning_is_additive {
            completion
        } else {
            completion.saturating_sub(reasoning)
        };
        Self {
            input: prompt.saturating_sub(cached),
            cache_read: cached,
            output,
            reasoning,
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
