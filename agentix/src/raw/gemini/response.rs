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
    #[serde(default)]
    pub candidates_token_count: u32,
    pub total_token_count: u32,
    /// Tokens served from cache (implicit or explicit context cache hit).
    /// Official docs: a SUBSET of `promptTokenCount` ("this is still the
    /// total effective prompt size … includes the cached content").
    #[serde(default)]
    pub cached_content_token_count: u32,
    /// Thinking tokens. Official docs: NOT included in
    /// `candidatesTokenCount` — `totalTokenCount = prompt + thoughts +
    /// candidates` — and billed at the output rate.
    #[serde(default)]
    pub thoughts_token_count: u32,
    /// Per-modality breakdown of the prompt (TEXT / IMAGE / AUDIO / VIDEO /
    /// DOCUMENT). The entries partition `promptTokenCount`.
    #[serde(default)]
    pub prompt_tokens_details: Vec<ModalityTokenCount>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ModalityTokenCount {
    #[serde(default)]
    pub modality: String,
    #[serde(default)]
    pub token_count: u32,
}

// Gemini billing semantics: billed input = (prompt − cached) × full rate +
// cached × cache rate; billed output = candidates + thoughts, both at the
// output rate (Pro-class models additionally tier both rates by prompt
// length — that lives in the price sheet, not here). Modality entries
// partition the prompt; cached tokens are attributed to the TEXT bucket
// (cached prefixes are overwhelmingly text — a documented approximation).
impl From<UsageMetadata> for crate::types::Usage {
    fn from(u: UsageMetadata) -> Self {
        let mut out = crate::types::Usage {
            cache_read: u.cached_content_token_count as u64,
            output: u.candidates_token_count as u64,
            reasoning: u.thoughts_token_count as u64,
            ..Default::default()
        };
        let mut text = u.prompt_token_count as u64;
        for m in &u.prompt_tokens_details {
            let n = m.token_count as u64;
            match m.modality.as_str() {
                "IMAGE" => out.input_image += n,
                "AUDIO" => out.input_audio += n,
                "VIDEO" => out.input_video += n,
                "DOCUMENT" => out.input_document += n,
                // TEXT / MODALITY_UNSPECIFIED stay in the text bucket.
                _ => continue,
            }
            text = text.saturating_sub(n);
        }
        out.input = text.saturating_sub(out.cache_read);
        out
    }
}
