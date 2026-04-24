//! Live smoke test for the OpenAI Responses API (requires OPENAI_API_KEY).
//!
//! Exercises the reasoning + tool_call round-trip that was the whole point
//! of switching `Provider::OpenAI` off Chat Completions. If the API ever
//! tightens its "reasoning item was provided without its required following
//! item" check, or if our `provider_data` splicing drifts out of spec, this
//! is the tripwire.
//!
//! Default-ignored — run with `cargo test -- --ignored`.

use agentix::msg::LlmEvent;
use agentix::{Content, Message, Provider, ReasoningEffort, Request, Tool as _, tool};
use futures::StreamExt;

/// Simple tool so the reasoning model has a reason to think + function_call
/// in the same turn.
#[tool]
/// Look up the capital of a country by name.
/// country: English country name, e.g. "France".
async fn capital_of(country: String) -> String {
    match country.to_lowercase().as_str() {
        "france" => "Paris".into(),
        "japan" => "Tokyo".into(),
        "brazil" => "Brasília".into(),
        _ => "Unknown".into(),
    }
}

#[tokio::test]
#[ignore]
async fn openai_responses_round_trip_with_tool_use() {
    let key = match std::env::var("OPENAI_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("OPENAI_API_KEY not set, skipping");
            return;
        }
    };

    let http = reqwest::Client::new();
    let tools = capital_of;

    // ── Round 1: ask a question that should trigger reasoning + tool call.
    let mut stream = Request::new(Provider::OpenAI, &key)
        .model("gpt-5")
        .reasoning_effort(ReasoningEffort::Medium)
        .user("What is the capital of France? Use the tool and then answer.")
        .tools(tools.raw_tools())
        .stream(&http)
        .await
        .expect("round 1 stream should open");

    let mut r1_content = String::new();
    let mut r1_reasoning = String::new();
    let mut r1_tool_calls: Vec<agentix::ToolCall> = Vec::new();
    let mut r1_state: Option<serde_json::Value> = None;
    while let Some(ev) = stream.next().await {
        match ev {
            LlmEvent::Token(t) => r1_content.push_str(&t),
            LlmEvent::Reasoning(r) => r1_reasoning.push_str(&r),
            LlmEvent::ToolCall(tc) => r1_tool_calls.push(tc),
            LlmEvent::AssistantState(v) => r1_state = Some(v),
            LlmEvent::Error(e) => panic!("round 1 error: {e}"),
            LlmEvent::Done => break,
            _ => {}
        }
    }
    assert!(
        !r1_tool_calls.is_empty(),
        "reasoning model should invoke the tool for this prompt"
    );
    assert!(
        r1_state.is_some(),
        "reasoning+tool turn must emit provider_data for round-trip"
    );

    // ── Round 2: feed R1 back (with the encrypted_content round-tripped via
    // provider_data) + a tool_result. Any splicing bug produces a 400 with
    // the diagnostic `'function_call' was provided without its required
    // 'reasoning' item`.
    let first_call = r1_tool_calls[0].clone();
    let history = vec![
        Message::User(vec![Content::text(
            "What is the capital of France? Use the tool and then answer.",
        )]),
        Message::Assistant {
            content: if r1_content.is_empty() {
                None
            } else {
                Some(r1_content.clone())
            },
            reasoning: if r1_reasoning.is_empty() {
                None
            } else {
                Some(r1_reasoning.clone())
            },
            tool_calls: r1_tool_calls.clone(),
            provider_data: r1_state.clone(),
        },
        Message::ToolResult {
            call_id: first_call.id,
            content: vec![Content::text("Paris")],
        },
    ];

    let mut stream = Request::new(Provider::OpenAI, &key)
        .model("gpt-5")
        .reasoning_effort(ReasoningEffort::Medium)
        .messages(history)
        .tools(tools.raw_tools())
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
        "round 2 should produce a non-empty answer"
    );
    eprintln!("r1 content: {r1_content}\nr1 reasoning: {r1_reasoning}\nr2: {r2_content}");
}
