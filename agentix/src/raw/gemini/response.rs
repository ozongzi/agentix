use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Response {
    pub candidates: Option<Vec<Candidate>>,
    pub usage_metadata: Option<UsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub content: ResponseContent,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ResponseContent {
    pub parts: Vec<ResponsePart>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResponsePart {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub function_call: Option<ResponseFunctionCall>,
    /// `true` distinguishes a summarized chain-of-thought part from an answer
    /// part. Only present when `includeThoughts: true` was requested.
    #[serde(default)]
    pub thought: Option<bool>,
    /// Encrypted hint the server validates on subsequent turns. Gemini 3
    /// enforces presence on the first `functionCall` part per step; older
    /// models attach it to the first part of any type. Must round-trip
    /// verbatim — we carry the entire part through `provider_data`.
    #[serde(default)]
    pub thought_signature: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ResponseFunctionCall {
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    pub prompt_token_count: u32,
    pub candidates_token_count: u32,
    pub total_token_count: u32,
    /// Tokens served from cache (implicit or explicit context cache hit).
    #[serde(default)]
    pub cached_content_token_count: u32,
    /// Tokens spent on internal thinking — available when
    /// `includeThoughts: true`.
    #[serde(default)]
    pub thoughts_token_count: u32,
}

impl From<UsageMetadata> for crate::types::UsageStats {
    fn from(u: UsageMetadata) -> Self {
        Self {
            prompt_tokens: u.prompt_token_count as usize,
            completion_tokens: u.candidates_token_count as usize,
            total_tokens: u.total_token_count as usize,
            cache_read_tokens: u.cached_content_token_count as usize,
            reasoning_tokens: u.thoughts_token_count as usize,
            ..Default::default()
        }
    }
}
