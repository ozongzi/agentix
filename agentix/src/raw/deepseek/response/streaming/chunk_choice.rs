use serde::Deserialize;

use super::delta::Delta;
use crate::raw::deepseek::response::non_streaming::{FinishReason, Logprobs};

#[derive(Debug, Deserialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    #[serde(default)]
    pub finish_reason: Option<FinishReason>,
    #[serde(default)]
    pub logprobs: Option<Logprobs>,
}
