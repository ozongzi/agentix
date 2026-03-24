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
    pub function_call: Option<ResponseFunctionCall>,
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
}

impl From<UsageMetadata> for crate::types::UsageStats {
    fn from(u: UsageMetadata) -> Self {
        Self {
            prompt_tokens:     u.prompt_token_count as usize,
            completion_tokens: u.candidates_token_count as usize,
            total_tokens:      u.total_token_count as usize,
        }
    }
}
