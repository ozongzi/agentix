pub mod request;
pub mod response;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::{StreamExt, stream::BoxStream};
use tracing::debug;

use crate::config::AgentConfig;
use crate::error::ApiError;
use crate::msg::LlmEvent;
use crate::provider::{Provider, PostConfig, post_streaming};
use crate::request::{Message, Request, ToolCall};
use crate::raw::shared::ToolDefinition;
use crate::types::{PartialToolCall, StreamBufs, ToolCallChunk, UsageStats};

use request::Request as AnthropicRequest;
use response::{ContentBlockDelta, ContentBlockStart, StreamEvent};

pub struct AnthropicProvider {
    token: String,
}

impl AnthropicProvider {
    pub fn new(token: impl Into<String>) -> Self {
        Self { token: token.into() }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    async fn stream(
        &self,
        http:     &reqwest::Client,
        config:   &AgentConfig,
        messages: &[Message],
        tools:    &[ToolDefinition],
    ) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
        let req = build_request(config, messages, tools);
        let url = format!("{}/v1/messages", config.base_url.trim_end_matches('/'));

        let resp = post_streaming(http, &url, &req, &self.token, &PostConfig {
            use_query_key:  false,
            auth_header:    Some("x-api-key"),
            extra_headers:  &[("anthropic-version", "2023-06-01")],
            max_retries:    config.max_retries,
            retry_delay_ms: config.retry_delay_ms,
        }).await?;

        let token = self.token.clone();
        let _ = token;

        Ok(async_stream::stream! {
            let mut bufs = StreamBufs::new();
            let mut sse  = resp.bytes_stream().eventsource();

            while let Some(ev_res) = sse.next().await {
                match ev_res {
                    Ok(ev) if ev.data == "[DONE]" => break,
                    Ok(ev) => {
                        match serde_json::from_str::<StreamEvent>(&ev.data) {
                            Ok(chunk) => {
                                for lev in parse_stream_event(chunk, &mut bufs) { yield lev; }
                            }
                            Err(e) => {
                                debug!(data = %ev.data, error = %e, "anthropic chunk parse failed");
                            }
                        }
                    }
                    Err(e) => { yield LlmEvent::Error(e.to_string()); break; }
                }
            }

            for tc in finalize(&mut bufs) { yield LlmEvent::ToolCall(tc); }
            yield LlmEvent::Done;
        }.boxed())
    }
}

fn build_request(config: &AgentConfig, messages: &[Message], tools: &[ToolDefinition]) -> AnthropicRequest {
    use crate::request::ToolChoice;
    let tool_choice = if tools.is_empty() { None } else { Some(ToolChoice::Auto) };
    AnthropicRequest::from(Request {
        system_message:  config.system_prompt.clone(),
        messages:        messages.to_vec(),
        model:           config.model.clone(),
        tools:           if tools.is_empty() { None } else { Some(tools.to_vec()) },
        tool_choice,
        stream:          true,
        temperature:     config.temperature,
        max_tokens:      config.max_tokens,
        response_format: None,
        extra_body:      if config.extra_body.is_empty() { None } else { Some(config.extra_body.clone()) },
    })
}

fn parse_stream_event(ev: StreamEvent, bufs: &mut StreamBufs) -> Vec<LlmEvent> {
    match ev {
        StreamEvent::MessageStart { message } => {
            if let Some(u) = message.usage {
                return vec![LlmEvent::Usage(UsageStats::from(u))];
            }
            vec![]
        }
        StreamEvent::MessageDelta { usage, .. } => {
            if let Some(u) = usage {
                return vec![LlmEvent::Usage(UsageStats::from(u))];
            }
            vec![]
        }
        StreamEvent::ContentBlockStart { index, content_block } => {
            let idx = index as usize;
            if bufs.tool_call_bufs.len() <= idx {
                bufs.tool_call_bufs.resize_with(idx + 1, || None);
            }
            match content_block {
                ContentBlockStart::ToolUse { id, name } => {
                    bufs.tool_call_bufs[idx] = Some(PartialToolCall {
                        id: id.clone(), name: name.clone(), arguments: String::new(),
                    });
                    vec![LlmEvent::ToolCallChunk(ToolCallChunk {
                        id, name, delta: String::new(), index,
                    })]
                }
                ContentBlockStart::Text { text } if !text.is_empty() => {
                    bufs.content_buf.push_str(&text);
                    vec![LlmEvent::Token(text)]
                }
                ContentBlockStart::Thinking { thinking } if !thinking.is_empty() => {
                    bufs.reasoning_buf.push_str(&thinking);
                    vec![LlmEvent::Reasoning(thinking)]
                }
                _ => vec![],
            }
        }
        StreamEvent::ContentBlockDelta { index, delta } => match delta {
            ContentBlockDelta::TextDelta { text } if !text.is_empty() => {
                bufs.content_buf.push_str(&text);
                vec![LlmEvent::Token(text)]
            }
            ContentBlockDelta::ThinkingDelta { thinking } if !thinking.is_empty() => {
                bufs.reasoning_buf.push_str(&thinking);
                vec![LlmEvent::Reasoning(thinking)]
            }
            ContentBlockDelta::InputJsonDelta { partial_json } if !partial_json.is_empty() => {
                let idx = index as usize;
                if let Some(Some(partial)) = bufs.tool_call_bufs.get_mut(idx) {
                    partial.arguments.push_str(&partial_json);
                    vec![LlmEvent::ToolCallChunk(ToolCallChunk {
                        id: partial.id.clone(), name: partial.name.clone(),
                        delta: partial_json, index,
                    })]
                } else { vec![] }
            }
            _ => vec![],
        },
        _ => vec![],
    }
}

fn finalize(bufs: &mut StreamBufs) -> Vec<ToolCall> {
    bufs.tool_call_bufs.drain(..).flatten().map(|p| ToolCall {
        id: p.id, name: p.name, arguments: p.arguments,
    }).collect()
}
