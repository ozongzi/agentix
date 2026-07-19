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

#[derive(Debug, Deserialize, Default)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
    /// Cache-write tokens (usage accounting is always on since 2026).
    #[serde(default)]
    pub cache_write_tokens: u32,
    #[serde(default)]
    pub audio_tokens: u32,
}

#[derive(Debug, Deserialize, Default)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: u32,
}

#[derive(Debug, Deserialize, Default)]
pub struct CostDetails {
    #[serde(default)]
    pub upstream_inference_cost: Option<f64>,
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
    /// Actual amount charged to the account, in OpenRouter credits (USD).
    /// Authoritative — no estimation needed when present.
    #[serde(default)]
    pub cost: Option<f64>,
    /// `upstream_inference_cost` is the upstream provider's own charge
    /// (meaningful for BYOK requests).
    #[serde(default)]
    pub cost_details: Option<CostDetails>,
}

// OpenRouter billing semantics: OpenAI-style counting — cached and
// cache-write tokens are subsets of `prompt_tokens`; reasoning is billed as
// output. Whether `completion_tokens` numerically includes reasoning is not
// documented, so we apply the same identity test as the Grok adapter.
// Cache pricing is passed through from the upstream with no markup, and
// `usage.cost` is the actual USD charge.
impl From<Usage> for crate::types::Usage {
    fn from(u: Usage) -> Self {
        let pd = u.prompt_tokens_details.unwrap_or_default();
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
            input: prompt
                .saturating_sub(pd.cached_tokens as u64)
                .saturating_sub(pd.cache_write_tokens as u64)
                .saturating_sub(pd.audio_tokens as u64),
            input_audio: pd.audio_tokens as u64,
            cache_read: pd.cached_tokens as u64,
            cache_write_5m: pd.cache_write_tokens as u64,
            output,
            reasoning,
            reported_cost: u.cost.map(|amount| crate::types::ReportedCost {
                amount,
                currency: "USD".to_string(),
            }),
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
