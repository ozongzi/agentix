//! Live smoke test for prompt-caching changes (requires OPENROUTER_API_KEY).

use agentix::{Provider, Request};
use agentix::msg::LlmEvent;
use futures::StreamExt;

#[tokio::test]
async fn openrouter_live_roundtrip() {
    let key = match std::env::var("OPENROUTER_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("OPENROUTER_API_KEY not set, skipping live test");
            return;
        }
    };

    let http = reqwest::Client::new();
    let mut stream = Request::new(Provider::OpenRouter, &key)
        .model("anthropic/claude-haiku-4-5")
        .system_prompt("You are a helpful assistant.")
        .user("Reply with exactly: OK")
        .stream(&http)
        .await
        .expect("stream should open");

    let mut text = String::new();
    while let Some(ev) = stream.next().await {
        match ev {
            LlmEvent::Token(t) => text.push_str(&t),
            LlmEvent::Error(e) => panic!("LLM error: {e}"),
            LlmEvent::Done => break,
            _ => {}
        }
    }
    assert!(!text.is_empty(), "should receive non-empty response, got: {text:?}");
    eprintln!("live response: {text}");
}
