use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub completion_tokens: u32,
    pub prompt_tokens: u32,
    #[serde(default)]
    pub prompt_cache_hit_tokens: Option<u32>,
    #[serde(default)]
    pub prompt_cache_miss_tokens: Option<u32>,
    pub total_tokens: u32,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    // Not mentioned in the API documentation
    // #[serde(default)]
    // pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: u32,
}

// #[derive(Debug, Deserialize)]
// pub struct PromptTokensDetails {
//     pub cached_tokens: u32,
// }
