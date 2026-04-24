//! Live smoke test for Gemini 3 thinking + `thoughtSignature` round-trip
//! (requires GOOGLE_API_KEY).
//!
//! Gemini 3 enforces `thoughtSignature` on the first `functionCall` part per
//! step — miss it and you get a 400. This smoke test exercises the full
//! round-trip to catch any drift in our `gemini_parts` splicing.
//!
//! Default-ignored — run with `cargo test -- --ignored`.

use agentix::msg::LlmEvent;
use agentix::{Content, Message, Provider, ReasoningEffort, Request, Tool as _, tool};
use futures::StreamExt;

#[tool]
/// Look up the capital of a country by name.
/// country: English country name, e.g. "France".
async fn capital_of(country: String) -> String {
    match country.to_lowercase().as_str() {
        "france" => "Paris".into(),
        "japan" => "Tokyo".into(),
        _ => "Unknown".into(),
    }
}

#[tokio::test]
#[ignore]
async fn gemini_3_thought_signature_round_trip() {
    let key = match std::env::var("GOOGLE_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("GOOGLE_API_KEY not set, skipping");
            return;
        }
    };

    let http = reqwest::Client::new();
    let tools = capital_of;

    let mut stream = Request::new(Provider::Gemini, &key)
        .model("gemini-3-pro")
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
        "gemini-3-pro should invoke the tool for this prompt"
    );
    assert!(
        r1_state.is_some(),
        "thinking+functionCall turn must emit provider_data"
    );

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

    let mut stream = Request::new(Provider::Gemini, &key)
        .model("gemini-3-pro")
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
        "round 2 should answer after receiving the tool result"
    );
    eprintln!("r1: {r1_content}\nr1 reasoning: {r1_reasoning}\nr2: {r2_content}");
}
