use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::{
    message::Message, model::Model, response_format::ResponseFormat, stop::Stop,
    stream_options::StreamOptions, thinking::Thinking, tool::Tool, tool_choice::ToolChoice,
};

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatCompletionRequest {
    /// List of messages in the conversation.
    pub messages: Vec<Message>,

    /// The model ID to use. Use `deepseek-chat` for faster responses or `deepseek-reasoner` for deeper reasoning capabilities.
    pub model: Model,

    /// Controls switching between reasoning (thinking) and non-reasoning modes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,

    /// Possible values: >= -2 and <= 2
    /// Default value: 0
    /// A number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text,
    /// reducing the chance of repeated content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Maximum number of tokens to generate for the completion in a single request.
    /// The combined length of input and output tokens is limited by the model's context window.
    /// See documentation for ranges and defaults.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Possible values: >= -2 and <= 2
    /// Default value: 0
    /// A number between -2.0 and 2.0. Positive values penalize new tokens if they already appear in the text,
    /// encouraging the model to introduce new topics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// An object specifying the format the model must output.
    /// Set to `{ "type": "json_object" }` to enable JSON mode which enforces valid JSON output.
    /// Note: When using JSON mode you must also instruct the model via system or user messages to output JSON.
    /// Otherwise the model may emit whitespace until token limits are reached which can appear to hang.
    /// Also, if `finish_reason == "length"`, the output may be truncated due to `max_tokens` or context limits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// A string or up to 16 strings. Generation will stop when one of these tokens is encountered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Stop>,

    /// If true, the response will be streamed as SSE (server-sent events). The stream ends with `data: [DONE]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Options related to streaming output. Only valid when `stream` is true.
    /// `include_usage`: boolean
    /// If true, an extra chunk with `usage` (aggregate token counts) will be sent before the final `data: [DONE]`.
    /// Other chunks also include `usage` but with a null value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Possible values: <= 2
    /// Default value: 1
    /// Sampling temperature between 0 and 2. Higher values (e.g. 0.8) produce more random output;
    /// lower values (e.g. 0.2) make output more focused and deterministic.
    /// Typically change either `temperature` or `top_p`, not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Possible values: <= 1
    /// Default value: 1
    /// An alternative to temperature that considers only the top `p` probability mass.
    /// For example, `top_p = 0.1` means only tokens comprising the top 10% probability mass are considered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// List of tools the model may call. Currently only `function` is supported.
    /// Provide a list of functions that accept JSON input. Up to 128 functions are supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// Controls how the model may call tools:
    /// - `none`: the model will not call tools and will produce a normal message.
    /// - `auto`: the model can choose to produce a message or call one or more tools.
    /// - `required`: the model must call one or more tools.
    ///
    /// Specifying a particular tool via `{"type":"function","function":{"name":"my_function"}}` forces the model to call that tool.
    ///
    /// Default is `none` when no tools exist; when tools exist the default is `auto`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// logprobs boolean NULLABLE
    /// Return log-probabilities for the output tokens. If true, logprobs for each output token are returned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,

    /// Possible values: <= 20
    /// An integer N between 0 and 20 that returns the top-N token log-probabilities for each output position.
    /// When specifying this parameter, `logprobs` must be true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    /// Extra arbitrary JSON body fields. When set, these key/value pairs are merged
    /// into the top-level request JSON. Use this to pass provider-specific or
    /// custom fields not yet modeled by the library.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<Map<String, Value>>,
}

impl ChatCompletionRequest {
    /// Add a single extra top-level field to the request body (in-place).
    ///
    /// This merges the given `key` / `value` pair into the request's
    /// `extra_body` map, creating that map if necessary. Values placed into
    /// `extra_body` are serialized into the top-level JSON of the request
    /// due to `#[serde(flatten)]`, so they appear as peers to fields such as
    /// `messages` and `model`.
    ///
    /// Notes:
    /// - Do not add keys that intentionally collide with existing top-level
    ///   fields (for example `messages` or `model`) unless you explicitly want
    ///   to override them — such collisions are not recommended.
    /// - Use this in-place helper when you have a mutable `ChatCompletionRequest`
    ///   instance and want to add a field without consuming the value.
    ///
    /// Example:
    /// ```
    /// # use agentix::raw::deepseek::request::ChatCompletionRequest;
    /// # use serde_json::json;
    /// let mut req = ChatCompletionRequest::default();
    /// req.add_extra_field("provider_opt", json!("x"));
    /// ```
    pub fn add_extra_field(&mut self, key: impl Into<String>, value: Value) {
        if let Some(ref mut m) = self.extra_body {
            m.insert(key.into(), value);
        } else {
            let mut m = Map::new();
            m.insert(key.into(), value);
            self.extra_body = Some(m);
        }
    }

    /// Builder-style helper to add a single extra field and return the owned
    /// request for chaining.
    ///
    /// This is a convenience that consumes (takes ownership of) `self`, adds
    /// the given key/value pair to `extra_body`, and returns the modified
    /// `ChatCompletionRequest` so you can continue chaining builder calls.
    ///
    /// Example:
    /// ```
    /// # use agentix::raw::deepseek::request::ChatCompletionRequest;
    /// # use serde_json::json;
    /// let req = ChatCompletionRequest::default()
    ///     .with_extra_field("provider_opt", json!("x"));
    /// ```
    pub fn with_extra_field(mut self, key: impl Into<String>, value: Value) -> Self {
        self.add_extra_field(key, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::deepseek::request::message::Role;

    #[test]
    fn test_chat_completion_request_serialization() {
        let request = ChatCompletionRequest {
            messages: vec![Message {
                role: Role::User,
                content: Some("Hello, world!".to_string()),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                reasoning_content: None,
                prefix: None,
            }],
            model: Model::DeepseekChat,
            thinking: None,
            frequency_penalty: Some(0.5),
            max_tokens: Some(100),
            presence_penalty: None,
            response_format: None,
            stop: None,
            stream: Some(false),
            stream_options: None,
            temperature: Some(0.7),
            top_p: None,
            tools: None,
            tool_choice: None,
            logprobs: None,
            top_logprobs: None,
            extra_body: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: ChatCompletionRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.messages.len(), 1);
        assert_eq!(
            parsed.messages[0].content.as_ref().unwrap(),
            "Hello, world!"
        );
        assert!(matches!(parsed.model, Model::DeepseekChat));
        assert_eq!(parsed.frequency_penalty, Some(0.5));
        assert_eq!(parsed.max_tokens, Some(100));
        assert_eq!(parsed.stream, Some(false));
        assert_eq!(parsed.temperature, Some(0.7));
    }

    #[test]
    fn test_default_chat_completion_request() {
        let request = ChatCompletionRequest::default();

        assert!(request.messages.is_empty());
        assert!(matches!(request.model, Model::DeepseekChat));
        assert!(request.thinking.is_none());
        assert!(request.frequency_penalty.is_none());
        assert!(request.max_tokens.is_none());
        assert!(request.presence_penalty.is_none());
        assert!(request.response_format.is_none());
        assert!(request.stop.is_none());
        assert!(request.stream.is_none());
        assert!(request.stream_options.is_none());
        assert!(request.temperature.is_none());
        assert!(request.top_p.is_none());
        assert!(request.tools.is_none());
        assert!(request.tool_choice.is_none());
        assert!(request.logprobs.is_none());
        assert!(request.top_logprobs.is_none());
        assert!(request.extra_body.is_none());
    }

    #[test]
    fn test_extra_body_serialize_merge() {
        use crate::raw::model::Model;
        use serde_json::{Map, Value, json};

        // Build an extra map
        let mut extra = Map::<String, Value>::new();
        extra.insert("x_custom".to_string(), json!("v1"));
        extra.insert("x_flag".to_string(), json!(true));

        // Create a request with extra_body set
        let req = ChatCompletionRequest {
            messages: vec![],
            model: Model::DeepseekChat,
            extra_body: Some(extra),
            ..Default::default()
        };

        // Serialize to a Value and assert the custom keys are present at top-level
        let v = serde_json::to_value(&req).expect("serialize");
        assert_eq!(
            v.get("x_custom").and_then(|val| val.as_str()).unwrap(),
            "v1"
        );
        assert!(v.get("x_flag").and_then(|val| val.as_bool()).unwrap());
    }
}
