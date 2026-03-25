use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::config::AgentConfig;
use crate::error::ApiError;
use crate::msg::LlmEvent;
use crate::provider::Provider;
use crate::request::Message;
use crate::raw::shared::ToolDefinition;

use crate::raw::openai::{stream_openai_compatible, complete_openai_compatible};
use crate::types::CompleteResponse;

/// Provider for the DeepSeek API.
pub struct DeepSeekProvider { token: String }

impl DeepSeekProvider {
    pub fn new(token: impl Into<String>) -> Self { Self { token: token.into() } }
}

#[async_trait]
impl Provider for DeepSeekProvider {
    async fn stream(
        &self,
        http:     &reqwest::Client,
        config:   &AgentConfig,
        messages: &[Message],
        tools:    &[ToolDefinition],
    ) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
        stream_openai_compatible(
            &self.token, http, config, messages, tools,
            Some(prepare_history),
        ).await
    }

    async fn complete(
        &self,
        http:     &reqwest::Client,
        config:   &AgentConfig,
        messages: &[Message],
        tools:    &[ToolDefinition],
    ) -> Result<CompleteResponse, ApiError> {
        complete_openai_compatible(
            &self.token, http, config, messages, tools,
            Some(prepare_history),
        ).await
    }
}

/// DeepSeek-reasoner history rules:
/// - assistant WITH tool_calls  → reasoning must be present (fill "" if absent)
/// - assistant WITHOUT tool_calls → reasoning must be None
fn prepare_history(messages: Vec<Message>) -> Vec<Message> {
    messages.into_iter().map(|m| match m {
        Message::Assistant { content, reasoning, tool_calls } => {
            let has_tools = !tool_calls.is_empty();
            Message::Assistant {
                content,
                reasoning: if has_tools { Some(reasoning.unwrap_or_default()) } else { None },
                tool_calls,
            }
        }
        other => other,
    }).collect()
}
