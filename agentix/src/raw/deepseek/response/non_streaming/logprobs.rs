use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Logprobs {
    #[serde(default)]
    pub content: Option<Vec<TokenLogprob>>,
    #[serde(default)]
    pub reasoning_content: Option<Vec<TokenLogprob>>,
}

#[derive(Debug, Deserialize)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f32,
    #[serde(default)]
    pub bytes: Option<Vec<u32>>,
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Debug, Deserialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    #[serde(default)]
    pub bytes: Option<Vec<u32>>,
}
