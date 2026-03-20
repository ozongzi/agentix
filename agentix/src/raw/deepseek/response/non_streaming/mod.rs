pub mod chat_completion_response;
pub mod choice;
pub mod finish_reason;
pub mod logprobs;
pub mod object_type;
pub mod usage;

pub use chat_completion_response::ChatCompletionResponse;
pub use choice::Choice;
pub use finish_reason::FinishReason;
pub use logprobs::{Logprobs, TokenLogprob, TopLogprob};
pub use object_type::ObjectType;
pub use usage::{CompletionTokensDetails, Usage};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_deserialization() {
        let json = r#"
            {
              "id": "dbdd2075-d78a-494a-afc9-b9ec5dc6bb64",
              "object": "chat.completion",
              "created": 1770982234,
              "model": "deepseek-chat",
              "choices": [
                {
                  "index": 0,
                  "message": {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today? 😊"
                  },
                  "logprobs": null,
                  "finish_reason": "stop"
                }
              ],
              "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 11,
                "total_tokens": 21,
                "prompt_tokens_details": {
                  "cached_tokens": 0
                },
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 10
              },
              "system_fingerprint": "fp_eaab8d114b_prod0820_fp8_kvcache"
            }
        "#;

        let response: ChatCompletionResponse = serde_json::from_str(json).unwrap();

        assert_eq!(
            response.id,
            "dbdd2075-d78a-494a-afc9-b9ec5dc6bb64".to_string()
        );
        assert!(matches!(response.object, ObjectType::ChatCompletion));
        assert_eq!(response.created, 1770982234);
        assert!(matches!(response.model, crate::raw::Model::DeepseekChat));
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].index, 0);
        assert!(matches!(
            response.choices[0].message.role,
            crate::raw::deepseek::request::message::Role::Assistant
        ));
        assert_eq!(
            response.choices[0].message.content.as_ref().unwrap(),
            "Hello! How can I assist you today? 😊"
        );
        assert!(response.choices[0].logprobs.is_none());
        assert!(matches!(
            response.choices[0].finish_reason,
            FinishReason::Stop
        ));

        assert_eq!(response.usage.prompt_tokens, 10);
        assert_eq!(response.usage.completion_tokens, 11);
        assert_eq!(response.usage.total_tokens, 21);
        assert_eq!(response.usage.prompt_cache_hit_tokens.unwrap(), 0);
        assert_eq!(response.usage.prompt_cache_miss_tokens.unwrap(), 10);
    }
}
