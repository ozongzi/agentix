//! Comprehensive test suite for `LlmClient`.
//!
//! Uses a mock `Provider` to test the full public API surface without real
//! network calls: constructors, config setters, clone semantics, snapshot,
//! stream dispatch, tool definition passthrough, and error propagation.

use std::sync::{Arc, Mutex};

use agentix::config::AgentConfig;
use agentix::error::ApiError;
use agentix::msg::LlmEvent;
use agentix::provider::Provider;
use agentix::raw::shared::{FunctionDefinition, ToolDefinition, ToolKind};
use agentix::request::{Message, ToolCall, UserContent};
use agentix::types::{ToolCallChunk, UsageStats};
use agentix::LlmClient;
use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use futures::StreamExt;
use serde_json::json;

// ── Mock Provider ─────────────────────────────────────────────────────────────

/// Records every call's arguments and returns a configurable sequence of events.
#[derive(Clone)]
struct MockProvider {
    /// Events to yield on each `stream()` call.
    events: Arc<Vec<LlmEvent>>,
    /// Captures (config_snapshot, messages_len, tools_len) from each call.
    calls: Arc<Mutex<Vec<MockCall>>>,
    /// Response to return from `complete()` calls.
    complete_response: Arc<agentix::CompleteResponse>,
    /// Captures args from each `complete()` call.
    complete_calls: Arc<Mutex<Vec<MockCall>>>,
}

#[derive(Debug, Clone)]
struct MockCall {
    config: AgentConfig,
    messages_len: usize,
    tools_len: usize,
}

impl MockProvider {
    fn new(events: Vec<LlmEvent>) -> Self {
        Self {
            events: Arc::new(events),
            calls: Arc::new(Mutex::new(Vec::new())),
            complete_response: Arc::new(agentix::CompleteResponse::default()),
            complete_calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn with_complete_response(mut self, resp: agentix::CompleteResponse) -> Self {
        self.complete_response = Arc::new(resp);
        self
    }

    fn calls(&self) -> Vec<MockCall> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl Provider for MockProvider {
    async fn stream(
        &self,
        _http: &reqwest::Client,
        config: &AgentConfig,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
        self.calls.lock().unwrap().push(MockCall {
            config: config.clone(),
            messages_len: messages.len(),
            tools_len: tools.len(),
        });
        let events = self.events.as_ref().clone();
        Ok(stream::iter(events).boxed())
    }

    async fn complete(
        &self,
        _http: &reqwest::Client,
        config: &AgentConfig,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<agentix::CompleteResponse, ApiError> {
        self.complete_calls.lock().unwrap().push(MockCall {
            config: config.clone(),
            messages_len: messages.len(),
            tools_len: tools.len(),
        });
        Ok(self.complete_response.as_ref().clone())
    }
}

/// A provider that always returns an error.
struct ErrorProvider {
    error_msg: String,
}

#[async_trait]
impl Provider for ErrorProvider {
    async fn stream(
        &self,
        _http: &reqwest::Client,
        _config: &AgentConfig,
        _messages: &[Message],
        _tools: &[ToolDefinition],
    ) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
        Err(ApiError::Other(self.error_msg.clone()))
    }

    async fn complete(
        &self,
        _http: &reqwest::Client,
        _config: &AgentConfig,
        _messages: &[Message],
        _tools: &[ToolDefinition],
    ) -> Result<agentix::CompleteResponse, ApiError> {
        Err(ApiError::Other(self.error_msg.clone()))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_tool_def(name: &str) -> ToolDefinition {
    ToolDefinition {
        kind: ToolKind::Function,
        function: FunctionDefinition {
            name: name.to_string(),
            description: Some(format!("A test tool named {name}")),
            parameters: json!({"type": "object", "properties": {}}),
            strict: None,
        },
    }
}

fn user_msg(text: &str) -> Message {
    Message::User(vec![UserContent::Text(text.to_string())])
}

async fn collect_events(client: &LlmClient, msgs: &[Message], tools: &[ToolDefinition]) -> Vec<LlmEvent> {
    let stream = client.stream(msgs, tools).await.expect("stream should succeed");
    stream.collect::<Vec<_>>().await
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CONSTRUCTOR TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod constructors {
    use super::*;

    #[test]
    fn deepseek_defaults() {
        let c = LlmClient::deepseek("sk-test");
        let snap = c.snapshot();
        assert_eq!(snap.base_url, "https://api.deepseek.com");
        assert_eq!(snap.model, "deepseek-chat");
        assert!(snap.system_prompt.is_none());
        assert!(snap.max_tokens.is_none());
        assert!(snap.temperature.is_none());
    }

    #[test]
    fn openai_defaults() {
        let c = LlmClient::openai("sk-test");
        let snap = c.snapshot();
        assert_eq!(snap.base_url, "https://api.openai.com/v1");
        assert_eq!(snap.model, "gpt-4o");
    }

    #[test]
    fn anthropic_defaults() {
        let c = LlmClient::anthropic("sk-test");
        let snap = c.snapshot();
        assert_eq!(snap.base_url, "https://api.anthropic.com");
        assert_eq!(snap.model, "claude-opus-4-5");
    }

    #[test]
    fn gemini_defaults() {
        let c = LlmClient::gemini("sk-test");
        let snap = c.snapshot();
        assert_eq!(snap.base_url, "https://generativelanguage.googleapis.com/v1beta");
        assert_eq!(snap.model, "gemini-2.0-flash");
    }

    #[test]
    fn from_parts_selects_correct_provider_and_overrides() {
        let c = LlmClient::from_parts("openai", "key", "https://custom.api", "custom-model");
        let snap = c.snapshot();
        assert_eq!(snap.base_url, "https://custom.api");
        assert_eq!(snap.model, "custom-model");
    }

    #[test]
    fn from_parts_unknown_provider_falls_back_to_deepseek() {
        let c = LlmClient::from_parts("unknown", "key", "https://x.com", "m1");
        let snap = c.snapshot();
        // base_url and model should be overridden regardless
        assert_eq!(snap.base_url, "https://x.com");
        assert_eq!(snap.model, "m1");
    }

    #[test]
    fn from_parts_all_providers() {
        for provider in ["deepseek", "openai", "anthropic", "gemini"] {
            let c = LlmClient::from_parts(provider, "key", "https://test", "test-model");
            let snap = c.snapshot();
            assert_eq!(snap.base_url, "https://test", "provider={provider}");
            assert_eq!(snap.model, "test-model", "provider={provider}");
        }
    }

    #[test]
    fn new_with_custom_provider() {
        let mock = MockProvider::new(vec![]);
        let config = AgentConfig {
            base_url: "https://mock.test".into(),
            model: "mock-v1".into(),
            ..Default::default()
        };
        let c = LlmClient::new(mock, config);
        let snap = c.snapshot();
        assert_eq!(snap.base_url, "https://mock.test");
        assert_eq!(snap.model, "mock-v1");
    }

    #[test]
    fn with_http_custom_client() {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        let mock = MockProvider::new(vec![]);
        let config = AgentConfig {
            base_url: "https://custom-http.test".into(),
            model: "m".into(),
            ..Default::default()
        };
        let c = LlmClient::with_http(mock, http, config);
        assert_eq!(c.snapshot().base_url, "https://custom-http.test");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CONFIG SETTER TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod config_setters {
    use super::*;

    #[test]
    fn model_setter() {
        let c = LlmClient::deepseek("k");
        c.model("new-model");
        assert_eq!(c.snapshot().model, "new-model");
    }

    #[test]
    fn base_url_setter() {
        let c = LlmClient::deepseek("k");
        c.base_url("https://new.url");
        assert_eq!(c.snapshot().base_url, "https://new.url");
    }

    #[test]
    fn system_prompt_setter() {
        let c = LlmClient::deepseek("k");
        assert!(c.snapshot().system_prompt.is_none());
        c.system_prompt("You are helpful.");
        assert_eq!(c.snapshot().system_prompt.unwrap(), "You are helpful.");
    }

    #[test]
    fn clear_system_prompt() {
        let c = LlmClient::deepseek("k");
        c.system_prompt("hello");
        assert!(c.snapshot().system_prompt.is_some());
        c.clear_system_prompt();
        assert!(c.snapshot().system_prompt.is_none());
    }

    #[test]
    fn max_tokens_setter() {
        let c = LlmClient::deepseek("k");
        assert!(c.snapshot().max_tokens.is_none());
        c.max_tokens(2048);
        assert_eq!(c.snapshot().max_tokens, Some(2048));
    }

    #[test]
    fn temperature_setter() {
        let c = LlmClient::deepseek("k");
        assert!(c.snapshot().temperature.is_none());
        c.temperature(0.7);
        assert_eq!(c.snapshot().temperature, Some(0.7));
    }

    #[test]
    fn chained_setters() {
        let c = LlmClient::deepseek("k");
        c.model("m1")
            .base_url("https://chain.test")
            .system_prompt("sp")
            .max_tokens(100)
            .temperature(0.5);
        let snap = c.snapshot();
        assert_eq!(snap.model, "m1");
        assert_eq!(snap.base_url, "https://chain.test");
        assert_eq!(snap.system_prompt.unwrap(), "sp");
        assert_eq!(snap.max_tokens, Some(100));
        assert_eq!(snap.temperature, Some(0.5));
    }

    #[test]
    fn setters_return_self_ref() {
        let c = LlmClient::deepseek("k");
        // Verify chaining works by using the returned reference
        let ret = c.model("m");
        ret.base_url("u");
        assert_eq!(c.snapshot().base_url, "u");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CLONE SEMANTICS
// ═══════════════════════════════════════════════════════════════════════════════

mod clone_semantics {
    use super::*;

    #[test]
    fn clones_share_config() {
        let c1 = LlmClient::deepseek("k");
        let c2 = c1.clone();
        c1.model("changed");
        assert_eq!(c2.snapshot().model, "changed");
    }

    #[test]
    fn clone_mutation_visible_both_ways() {
        let c1 = LlmClient::deepseek("k");
        let c2 = c1.clone();
        c2.system_prompt("from c2");
        assert_eq!(c1.snapshot().system_prompt.unwrap(), "from c2");

        c1.max_tokens(999);
        assert_eq!(c2.snapshot().max_tokens, Some(999));
    }

    #[test]
    fn multiple_clones_all_share() {
        let c = LlmClient::deepseek("k");
        let clones: Vec<_> = (0..10).map(|_| c.clone()).collect();
        c.temperature(0.42);
        for (i, cl) in clones.iter().enumerate() {
            assert_eq!(cl.snapshot().temperature, Some(0.42), "clone {i} should see temperature");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  STREAM TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod stream_tests {
    use super::*;

    #[tokio::test]
    async fn stream_yields_tokens() {
        let mock = MockProvider::new(vec![
            LlmEvent::Token("Hello".into()),
            LlmEvent::Token(" world".into()),
            LlmEvent::Done,
        ]);
        let client = LlmClient::new(mock, AgentConfig::default());
        let events = collect_events(&client, &[user_msg("hi")], &[]).await;

        let tokens: Vec<_> = events
            .iter()
            .filter_map(|e| if let LlmEvent::Token(t) = e { Some(t.as_str()) } else { None })
            .collect();
        assert_eq!(tokens, vec!["Hello", " world"]);
        assert!(matches!(events.last(), Some(LlmEvent::Done)));
    }

    #[tokio::test]
    async fn stream_yields_reasoning() {
        let mock = MockProvider::new(vec![
            LlmEvent::Reasoning("thinking...".into()),
            LlmEvent::Token("answer".into()),
            LlmEvent::Done,
        ]);
        let client = LlmClient::new(mock, AgentConfig::default());
        let events = collect_events(&client, &[user_msg("q")], &[]).await;

        assert!(matches!(&events[0], LlmEvent::Reasoning(r) if r == "thinking..."));
        assert!(matches!(&events[1], LlmEvent::Token(t) if t == "answer"));
    }

    #[tokio::test]
    async fn stream_yields_tool_calls() {
        let tc = ToolCall {
            id: "call_1".into(),
            name: "get_weather".into(),
            arguments: r#"{"city":"Beijing"}"#.into(),
        };
        let chunk = ToolCallChunk {
            id: "call_1".into(),
            name: "get_weather".into(),
            delta: r#"{"city"#.into(),
            index: 0,
        };
        let mock = MockProvider::new(vec![
            LlmEvent::ToolCallChunk(chunk),
            LlmEvent::ToolCall(tc),
            LlmEvent::Done,
        ]);
        let client = LlmClient::new(mock, AgentConfig::default());
        let events = collect_events(&client, &[user_msg("weather")], &[]).await;

        assert!(matches!(&events[0], LlmEvent::ToolCallChunk(c) if c.name == "get_weather"));
        assert!(matches!(&events[1], LlmEvent::ToolCall(tc) if tc.id == "call_1" && tc.name == "get_weather"));
    }

    #[tokio::test]
    async fn stream_yields_usage() {
        let usage = UsageStats {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        };
        let mock = MockProvider::new(vec![
            LlmEvent::Token("hi".into()),
            LlmEvent::Usage(usage),
            LlmEvent::Done,
        ]);
        let client = LlmClient::new(mock, AgentConfig::default());
        let events = collect_events(&client, &[user_msg("q")], &[]).await;

        let usage_ev = events.iter().find(|e| matches!(e, LlmEvent::Usage(_)));
        assert!(usage_ev.is_some());
        if let Some(LlmEvent::Usage(u)) = usage_ev {
            assert_eq!(u.prompt_tokens, 10);
            assert_eq!(u.completion_tokens, 20);
            assert_eq!(u.total_tokens, 30);
        }
    }

    #[tokio::test]
    async fn stream_yields_error_event() {
        let mock = MockProvider::new(vec![
            LlmEvent::Error("something went wrong".into()),
            LlmEvent::Done,
        ]);
        let client = LlmClient::new(mock, AgentConfig::default());
        let events = collect_events(&client, &[user_msg("q")], &[]).await;

        assert!(matches!(&events[0], LlmEvent::Error(e) if e == "something went wrong"));
    }

    #[tokio::test]
    async fn stream_empty_events() {
        let mock = MockProvider::new(vec![]);
        let client = LlmClient::new(mock, AgentConfig::default());
        let events = collect_events(&client, &[user_msg("q")], &[]).await;
        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn stream_complex_conversation() {
        let mock = MockProvider::new(vec![
            LlmEvent::Token("I'll help.".into()),
            LlmEvent::Done,
        ]);
        let client = LlmClient::new(mock, AgentConfig::default());

        let messages = vec![
            user_msg("What is 2+2?"),
            Message::Assistant {
                content: Some("4".into()),
                reasoning: None,
                tool_calls: vec![],
            },
            user_msg("And 3+3?"),
        ];
        let events = collect_events(&client, &messages, &[]).await;
        assert!(!events.is_empty());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CONFIG PASSTHROUGH TO PROVIDER
// ═══════════════════════════════════════════════════════════════════════════════

mod config_passthrough {
    use super::*;

    #[tokio::test]
    async fn config_snapshot_passed_to_provider() {
        let mock = MockProvider::new(vec![LlmEvent::Done]);
        let config = AgentConfig {
            base_url: "https://test.api".into(),
            model: "test-model".into(),
            system_prompt: Some("You are test.".into()),
            max_tokens: Some(512),
            temperature: Some(0.3),
            ..Default::default()
        };
        let client = LlmClient::new(mock.clone(), config);
        let _ = collect_events(&client, &[user_msg("hi")], &[]).await;

        let calls = mock.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].config.base_url, "https://test.api");
        assert_eq!(calls[0].config.model, "test-model");
        assert_eq!(calls[0].config.system_prompt.as_deref(), Some("You are test."));
        assert_eq!(calls[0].config.max_tokens, Some(512));
        assert_eq!(calls[0].config.temperature, Some(0.3));
    }

    #[tokio::test]
    async fn config_changes_visible_on_next_call() {
        let mock = MockProvider::new(vec![LlmEvent::Done]);
        let client = LlmClient::new(mock.clone(), AgentConfig::default());

        // First call with defaults
        let _ = collect_events(&client, &[user_msg("1")], &[]).await;

        // Mutate config
        client.model("new-model");
        client.system_prompt("new prompt");

        // Second call should see new config
        let _ = collect_events(&client, &[user_msg("2")], &[]).await;

        let calls = mock.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].config.model, "new-model");
        assert_eq!(calls[1].config.system_prompt.as_deref(), Some("new prompt"));
    }

    #[tokio::test]
    async fn messages_count_passed_through() {
        let mock = MockProvider::new(vec![LlmEvent::Done]);
        let client = LlmClient::new(mock.clone(), AgentConfig::default());

        let msgs = vec![user_msg("a"), user_msg("b"), user_msg("c")];
        let _ = collect_events(&client, &msgs, &[]).await;

        assert_eq!(mock.calls()[0].messages_len, 3);
    }

    #[tokio::test]
    async fn tool_definitions_passed_through() {
        let mock = MockProvider::new(vec![LlmEvent::Done]);
        let client = LlmClient::new(mock.clone(), AgentConfig::default());

        let tools = vec![make_tool_def("tool_a"), make_tool_def("tool_b")];
        let _ = collect_events(&client, &[user_msg("hi")], &tools).await;

        assert_eq!(mock.calls()[0].tools_len, 2);
    }

    #[tokio::test]
    async fn empty_tools_and_messages() {
        let mock = MockProvider::new(vec![LlmEvent::Done]);
        let client = LlmClient::new(mock.clone(), AgentConfig::default());

        let _ = collect_events(&client, &[], &[]).await;
        assert_eq!(mock.calls()[0].messages_len, 0);
        assert_eq!(mock.calls()[0].tools_len, 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  ERROR HANDLING
// ═══════════════════════════════════════════════════════════════════════════════

mod error_handling {
    use super::*;

    #[tokio::test]
    async fn provider_error_propagates() {
        let provider = ErrorProvider {
            error_msg: "mock provider error".into(),
        };
        let client = LlmClient::new(provider, AgentConfig::default());
        let result = client.stream(&[user_msg("hi")], &[]).await;

        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected error"),
        };
        assert!(err.to_string().contains("mock provider error"));
    }

    #[tokio::test]
    async fn different_error_types() {
        // Test ApiError::Other
        let provider = ErrorProvider { error_msg: "test error".into() };
        let client = LlmClient::new(provider, AgentConfig::default());
        let err = match client.stream(&[], &[]).await {
            Err(e) => e,
            Ok(_) => panic!("expected error"),
        };
        assert!(matches!(err, ApiError::Other(_)));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CONCURRENT ACCESS
// ═══════════════════════════════════════════════════════════════════════════════

mod concurrent {
    use super::*;

    #[tokio::test]
    async fn concurrent_config_mutations() {
        let client = LlmClient::deepseek("k");
        let mut handles = Vec::new();
        for i in 0..20 {
            let c = client.clone();
            handles.push(tokio::spawn(async move {
                c.model(format!("model-{i}"));
                c.temperature(i as f32 * 0.1);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        // No panic / deadlock — config is in a consistent state
        let snap = client.snapshot();
        assert!(!snap.model.is_empty());
    }

    #[tokio::test]
    async fn concurrent_stream_calls() {
        let mock = MockProvider::new(vec![
            LlmEvent::Token("ok".into()),
            LlmEvent::Done,
        ]);
        let client = LlmClient::new(mock.clone(), AgentConfig::default());

        let mut handles = Vec::new();
        for _ in 0..10 {
            let c = client.clone();
            handles.push(tokio::spawn(async move {
                let events = collect_events(&c, &[user_msg("hi")], &[]).await;
                assert!(!events.is_empty());
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(mock.calls().len(), 10);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SNAPSHOT ISOLATION
// ═══════════════════════════════════════════════════════════════════════════════

mod snapshot {
    use super::*;

    #[test]
    fn snapshot_is_a_copy() {
        let c = LlmClient::deepseek("k");
        let snap1 = c.snapshot();
        c.model("changed");
        let snap2 = c.snapshot();

        assert_eq!(snap1.model, "deepseek-chat");
        assert_eq!(snap2.model, "changed");
    }

    #[test]
    fn snapshot_captures_all_fields() {
        let c = LlmClient::deepseek("k");
        c.model("m").base_url("u").system_prompt("sp").max_tokens(42).temperature(0.9);
        let snap = c.snapshot();

        assert_eq!(snap.model, "m");
        assert_eq!(snap.base_url, "u");
        assert_eq!(snap.system_prompt, Some("sp".into()));
        assert_eq!(snap.max_tokens, Some(42));
        assert_eq!(snap.temperature, Some(0.9));
    }

    #[test]
    fn default_config_fields() {
        let config = AgentConfig::default();
        assert_eq!(config.base_url, "");
        assert_eq!(config.model, "");
        assert!(config.system_prompt.is_none());
        assert!(config.max_tokens.is_none());
        assert!(config.temperature.is_none());
        assert!(config.extra_body.is_empty());
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_delay_ms, 1000);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MULTI-TURN STREAM SEQUENCES
// ═══════════════════════════════════════════════════════════════════════════════

mod multi_turn {
    use super::*;

    #[tokio::test]
    async fn sequential_calls_get_independent_streams() {
        // Each call to stream() gets its own independent stream from the provider
        let mock = MockProvider::new(vec![
            LlmEvent::Token("response".into()),
            LlmEvent::Done,
        ]);
        let client = LlmClient::new(mock.clone(), AgentConfig::default());

        let events1 = collect_events(&client, &[user_msg("q1")], &[]).await;
        let events2 = collect_events(&client, &[user_msg("q2")], &[]).await;

        assert_eq!(events1.len(), events2.len());
        assert_eq!(mock.calls().len(), 2);
        assert_eq!(mock.calls()[0].messages_len, 1);
        assert_eq!(mock.calls()[1].messages_len, 1);
    }

    #[tokio::test]
    async fn growing_message_history() {
        let mock = MockProvider::new(vec![LlmEvent::Done]);
        let client = LlmClient::new(mock.clone(), AgentConfig::default());

        for i in 1..=5 {
            let msgs: Vec<_> = (0..i).map(|j| user_msg(&format!("msg-{j}"))).collect();
            let _ = collect_events(&client, &msgs, &[]).await;
        }

        let calls = mock.calls();
        assert_eq!(calls.len(), 5);
        for (i, call) in calls.iter().enumerate() {
            assert_eq!(call.messages_len, i + 1);
        }
    }

    #[tokio::test]
    async fn tool_result_in_history() {
        let mock = MockProvider::new(vec![LlmEvent::Done]);
        let client = LlmClient::new(mock.clone(), AgentConfig::default());

        let messages = vec![
            user_msg("search for X"),
            Message::Assistant {
                content: None,
                reasoning: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "search".into(),
                    arguments: r#"{"q":"X"}"#.into(),
                }],
            },
            Message::ToolResult {
                call_id: "call_1".into(),
                content: "found X".into(),
            },
        ];
        let _ = collect_events(&client, &messages, &[]).await;
        assert_eq!(mock.calls()[0].messages_len, 3);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  FULL EVENT SEQUENCE
// ═══════════════════════════════════════════════════════════════════════════════

mod full_event_sequence {
    use super::*;

    #[tokio::test]
    async fn realistic_turn_with_reasoning_and_tool_call() {
        let tc = ToolCall {
            id: "tc_1".into(),
            name: "calculator".into(),
            arguments: r#"{"expr":"2+2"}"#.into(),
        };
        let chunk = ToolCallChunk {
            id: "tc_1".into(),
            name: "calculator".into(),
            delta: r#"{"expr":"2+2"}"#.into(),
            index: 0,
        };
        let usage = UsageStats {
            prompt_tokens: 50,
            completion_tokens: 30,
            total_tokens: 80,
        };
        let mock = MockProvider::new(vec![
            LlmEvent::Reasoning("Let me think...".into()),
            LlmEvent::ToolCallChunk(chunk),
            LlmEvent::ToolCall(tc),
            LlmEvent::Usage(usage),
            LlmEvent::Done,
        ]);
        let client = LlmClient::new(mock, AgentConfig::default());
        let events = collect_events(&client, &[user_msg("2+2?")], &[make_tool_def("calculator")]).await;

        assert_eq!(events.len(), 5);
        assert!(matches!(&events[0], LlmEvent::Reasoning(_)));
        assert!(matches!(&events[1], LlmEvent::ToolCallChunk(_)));
        assert!(matches!(&events[2], LlmEvent::ToolCall(_)));
        assert!(matches!(&events[3], LlmEvent::Usage(_)));
        assert!(matches!(&events[4], LlmEvent::Done));
    }

    #[tokio::test]
    async fn text_only_response() {
        let mock = MockProvider::new(vec![
            LlmEvent::Token("The ".into()),
            LlmEvent::Token("answer ".into()),
            LlmEvent::Token("is 42.".into()),
            LlmEvent::Usage(UsageStats {
                prompt_tokens: 5,
                completion_tokens: 4,
                total_tokens: 9,
            }),
            LlmEvent::Done,
        ]);
        let client = LlmClient::new(mock, AgentConfig::default());
        let events = collect_events(&client, &[user_msg("what?")], &[]).await;

        let full_text: String = events
            .iter()
            .filter_map(|e| if let LlmEvent::Token(t) = e { Some(t.as_str()) } else { None })
            .collect();
        assert_eq!(full_text, "The answer is 42.");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  COMPLETE (NON-STREAMING) TESTS
// ═══════════════════════════════════════════════════════════════════════════════

mod complete_tests {
    use super::*;

    #[tokio::test]
    async fn complete_returns_content() {
        let mock = MockProvider::new(vec![]).with_complete_response(agentix::CompleteResponse {
            content: Some("Paris".into()),
            reasoning: None,
            tool_calls: vec![],
            usage: UsageStats::default(),
        });
        let client = LlmClient::new(mock, AgentConfig::default());
        let resp = client.complete(&[user_msg("capital of France?")], &[]).await.unwrap();
        assert_eq!(resp.content.as_deref(), Some("Paris"));
        assert!(resp.tool_calls.is_empty());
    }

    #[tokio::test]
    async fn complete_returns_reasoning() {
        let mock = MockProvider::new(vec![]).with_complete_response(agentix::CompleteResponse {
            content: Some("42".into()),
            reasoning: Some("Let me think step by step...".into()),
            tool_calls: vec![],
            usage: UsageStats::default(),
        });
        let client = LlmClient::new(mock, AgentConfig::default());
        let resp = client.complete(&[user_msg("meaning of life")], &[]).await.unwrap();
        assert_eq!(resp.reasoning.as_deref(), Some("Let me think step by step..."));
    }

    #[tokio::test]
    async fn complete_returns_tool_calls() {
        let mock = MockProvider::new(vec![]).with_complete_response(agentix::CompleteResponse {
            content: None,
            reasoning: None,
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: r#"{"city":"Beijing"}"#.into(),
            }],
            usage: UsageStats::default(),
        });
        let client = LlmClient::new(mock, AgentConfig::default());
        let tools = vec![make_tool_def("get_weather")];
        let resp = client.complete(&[user_msg("weather?")], &tools).await.unwrap();
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "get_weather");
        assert_eq!(resp.tool_calls[0].id, "call_1");
    }

    #[tokio::test]
    async fn complete_returns_usage() {
        let mock = MockProvider::new(vec![]).with_complete_response(agentix::CompleteResponse {
            content: Some("ok".into()),
            reasoning: None,
            tool_calls: vec![],
            usage: UsageStats { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
        });
        let client = LlmClient::new(mock, AgentConfig::default());
        let resp = client.complete(&[user_msg("hi")], &[]).await.unwrap();
        assert_eq!(resp.usage.prompt_tokens, 10);
        assert_eq!(resp.usage.completion_tokens, 5);
        assert_eq!(resp.usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn complete_passes_config_and_args() {
        let mock = MockProvider::new(vec![]);
        let config = AgentConfig {
            base_url: "https://test.api".into(),
            model: "test-model".into(),
            system_prompt: Some("You are test.".into()),
            ..Default::default()
        };
        let client = LlmClient::new(mock.clone(), config);
        let tools = vec![make_tool_def("t1"), make_tool_def("t2")];
        let _ = client.complete(&[user_msg("q1"), user_msg("q2")], &tools).await.unwrap();

        let calls = mock.complete_calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].config.base_url, "https://test.api");
        assert_eq!(calls[0].config.model, "test-model");
        assert_eq!(calls[0].config.system_prompt.as_deref(), Some("You are test."));
        assert_eq!(calls[0].messages_len, 2);
        assert_eq!(calls[0].tools_len, 2);
    }

    #[tokio::test]
    async fn complete_error_propagates() {
        let provider = ErrorProvider { error_msg: "complete failed".into() };
        let client = LlmClient::new(provider, AgentConfig::default());
        let err = client.complete(&[user_msg("hi")], &[]).await.unwrap_err();
        assert!(err.to_string().contains("complete failed"));
    }

    #[tokio::test]
    async fn complete_empty_response() {
        let mock = MockProvider::new(vec![]);
        let client = LlmClient::new(mock, AgentConfig::default());
        let resp = client.complete(&[], &[]).await.unwrap();
        assert!(resp.content.is_none());
        assert!(resp.reasoning.is_none());
        assert!(resp.tool_calls.is_empty());
    }

    #[tokio::test]
    async fn complete_config_changes_visible() {
        let mock = MockProvider::new(vec![]);
        let client = LlmClient::new(mock.clone(), AgentConfig::default());

        let _ = client.complete(&[user_msg("1")], &[]).await.unwrap();
        client.model("new-model");
        let _ = client.complete(&[user_msg("2")], &[]).await.unwrap();

        let calls = mock.complete_calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].config.model, "new-model");
    }
}
