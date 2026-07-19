use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Response {
    pub content: Vec<ResponseBlock>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub stop_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    RedactedThinking {
        data: String,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    /// Per-TTL breakdown of cache writes (newer API versions). When present,
    /// its parts sum to `cache_creation_input_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation: Option<CacheCreation>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct CacheCreation {
    #[serde(default)]
    pub ephemeral_5m_input_tokens: u32,
    #[serde(default)]
    pub ephemeral_1h_input_tokens: u32,
}

// Anthropic billing semantics: `input_tokens` already EXCLUDES cache tokens
// (total input = input + cache_read + cache_creation), so the wire fields map
// straight onto the disjoint buckets. Thinking tokens are billed as ordinary
// output and are not broken out on the wire, so `reasoning` stays 0.
impl From<Usage> for crate::types::Usage {
    fn from(u: Usage) -> Self {
        // Prefer the per-TTL breakdown; fall back to attributing all writes
        // to the 5-minute bucket (the default TTL, 1.25× write rate).
        let (write_5m, write_1h) = match u.cache_creation {
            Some(c) if c.ephemeral_5m_input_tokens + c.ephemeral_1h_input_tokens > 0 => {
                (c.ephemeral_5m_input_tokens, c.ephemeral_1h_input_tokens)
            }
            _ => (u.cache_creation_input_tokens, 0),
        };
        Self {
            input: u.input_tokens as u64,
            cache_read: u.cache_read_input_tokens as u64,
            cache_write_5m: write_5m as u64,
            cache_write_1h: write_1h as u64,
            output: u.output_tokens as u64,
            ..Default::default()
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart {
        message: MessageStart,
    },
    ContentBlockStart {
        index: u32,
        content_block: ContentBlockStart,
    },
    ContentBlockDelta {
        index: u32,
        delta: ContentBlockDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: MessageDelta,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        usage: Option<Usage>,
    },
    MessageStop,
    Error {
        error: StreamError,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MessageStart {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockStart {
    Text { text: String },
    ToolUse { id: String, name: String },
    Thinking { thinking: String },
    RedactedThinking { data: String },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MessageDelta {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub stop_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamError {
    pub r#type: String,
    pub message: String,
}
