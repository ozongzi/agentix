use serde::Deserialize;

use super::{finish_reason::FinishReason, logprobs::Logprobs};
use crate::raw::deepseek::request::message::Message;

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub finish_reason: FinishReason,
    pub index: u32,
    pub message: Message,
    #[serde(default)]
    pub logprobs: Option<Logprobs>,
}
