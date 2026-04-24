//! Live multi-turn smoke test for DeepSeek (requires DEEPSEEK_API_KEY).
//!
//! Targets the path changed when `prepare_history` was removed: an assistant
//! turn that carries `reasoning_content` but no tool calls is now sent back to
//! the API verbatim instead of stripped. Per the docs the API ignores
//! reasoning_content on non-tool-call turns, so this acts as a tripwire for
//! the case where DeepSeek tightens that rule.
//!
//! Default-ignored — run with `cargo test -- --ignored`.

use agentix::msg::LlmEvent;
use agentix::{Content, Message, Provider, ReasoningEffort, Request, ToolCall};
use futures::StreamExt;

#[tokio::test]
#[ignore]
async fn deepseek_reasoner_multi_turn_replays_reasoning() {
    let key = match std::env::var("DEEPSEEK_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("DEEPSEEK_API_KEY not set, skipping");
            return;
        }
    };

    let http = reqwest::Client::new();

    // Round 1 — deepseek-v4-pro produces reasoning_content alongside the answer.
    let mut stream = Request::new(Provider::DeepSeek, &key)
        .model("deepseek-v4-pro")
        .user("What is 2+3? Reply with just the number.")
        .stream(&http)
        .await
        .expect("round 1 stream should open");

    let mut r1_content = String::new();
    let mut r1_reasoning = String::new();
    let mut r1_tool_calls: Vec<ToolCall> = Vec::new();
    while let Some(ev) = stream.next().await {
        match ev {
            LlmEvent::Token(t) => r1_content.push_str(&t),
            LlmEvent::Reasoning(r) => r1_reasoning.push_str(&r),
            LlmEvent::ToolCall(tc) => r1_tool_calls.push(tc),
            LlmEvent::Error(e) => panic!("round 1 error: {e}"),
            LlmEvent::Done => break,
            _ => {}
        }
    }
    assert!(!r1_content.is_empty(), "round 1 should return content");
    assert!(
        !r1_reasoning.is_empty(),
        "deepseek-v4-pro should emit reasoning_content"
    );
    assert!(
        r1_tool_calls.is_empty(),
        "round 1 should not invoke tools (this test exercises the no-tool-call path)"
    );

    // Round 2 — replay R1's assistant turn with reasoning intact, then ask a
    // follow-up. If DeepSeek ever rejects reasoning_content on a non-tool-call
    // assistant turn, this round will error.
    let history = vec![
        Message::User(vec![Content::text(
            "What is 2+3? Reply with just the number.",
        )]),
        Message::Assistant {
            content: Some(r1_content.clone()),
            reasoning: Some(r1_reasoning),
            tool_calls: Vec::new(),
            provider_data: None,
        },
        Message::User(vec![Content::text(
            "Now add 10. Reply with just the number.",
        )]),
    ];

    let mut stream = Request::new(Provider::DeepSeek, &key)
        .model("deepseek-v4-pro")
        .messages(history)
        .stream(&http)
        .await
        .expect("round 2 stream should open");

    let mut r2_content = String::new();
    while let Some(ev) = stream.next().await {
        match ev {
            LlmEvent::Token(t) => r2_content.push_str(&t),
            LlmEvent::Error(e) => panic!("round 2 error: {e}"),
            LlmEvent::Done => break,
            _ => {}
        }
    }
    assert!(
        !r2_content.is_empty(),
        "round 2 should return non-empty content"
    );
    eprintln!("r1: {r1_content}  r2: {r2_content}");
}

/// Exercises the Standard variant of the DeepSeek `ReasoningMode` sum type:
/// thinking explicitly disabled, temperature passes through, no reasoning is
/// emitted. Verifies that the sampling-params branch reaches the API cleanly.
#[tokio::test]
#[ignore]
async fn deepseek_thinking_disabled_accepts_temperature() {
    let key = match std::env::var("DEEPSEEK_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("DEEPSEEK_API_KEY not set, skipping");
            return;
        }
    };

    let http = reqwest::Client::new();
    let mut stream = Request::new(Provider::DeepSeek, &key)
        .model("deepseek-v4-pro")
        .reasoning_effort(ReasoningEffort::None)
        .temperature(0.2)
        .user("Reply with exactly the word: ok")
        .stream(&http)
        .await
        .expect("stream should open");

    let mut content = String::new();
    let mut reasoning = String::new();
    while let Some(ev) = stream.next().await {
        match ev {
            LlmEvent::Token(t) => content.push_str(&t),
            LlmEvent::Reasoning(r) => reasoning.push_str(&r),
            LlmEvent::Error(e) => panic!("thinking-disabled request errored: {e}"),
            LlmEvent::Done => break,
            _ => {}
        }
    }
    assert!(!content.is_empty(), "should return non-empty content");
    assert!(
        reasoning.is_empty(),
        "thinking=disabled should not emit reasoning, got: {reasoning:?}"
    );
    eprintln!("thinking-disabled response: {content}");
}
