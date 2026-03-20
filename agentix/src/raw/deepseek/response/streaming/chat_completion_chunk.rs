use serde::Deserialize;

use super::{chunk_choice::ChunkChoice, chunk_object_type::ChunkObjectType};
use crate::raw::deepseek::response::non_streaming::Usage;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub choices: Vec<ChunkChoice>,
    pub created: u64,
    pub model: String,
    #[serde(default)]
    pub system_fingerprint: Option<String>,
    #[serde(rename = "object")]
    pub object: ChunkObjectType,
    #[serde(default)]
    pub usage: Option<Usage>,
}
