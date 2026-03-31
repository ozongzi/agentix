//! Tests for `agent_turns` and `AgentTurnsStream`.
//!
//! Uses mock streams constructed via `AgentTurnsStream::from_items` —
//! no real LLM calls are made.

use agentix::{AgentTurnsStream, ApiError, CompleteResponse, Provider, Request, ToolBundle};
use futures::StreamExt;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn resp(content: &str) -> CompleteResponse {
    CompleteResponse {
        content: Some(content.to_string()),
        ..Default::default()
    }
}

fn resp_no_content() -> CompleteResponse {
    CompleteResponse { content: None, ..Default::default() }
}

fn mock_err() -> ApiError {
    ApiError::Other("mock error".into())
}

// ── agent_turns() type tests ──────────────────────────────────────────────────

#[test]
fn agent_turns_returns_agent_turns_stream() {
    // Verify agent_turns() compiles and returns AgentTurnsStream.
    let _s = agentix::agent_turns(
        ToolBundle::default(),
        reqwest::Client::new(),
        Request::new(Provider::OpenAI, "sk-test"),
        vec![],
        None,
    );
}

#[test]
fn agent_turns_stream_implements_stream() {
    fn assert_stream<S: futures::Stream>(_: &S) {}
    let s = agentix::agent_turns(
        ToolBundle::default(),
        reqwest::Client::new(),
        Request::new(Provider::OpenAI, "sk-test"),
        vec![],
        None,
    );
    assert_stream(&s);
}

// ── last_ok() ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn last_ok_returns_last_successful_item() {
    let s = AgentTurnsStream::from_items(vec![
        Ok(resp("turn 1")),
        Ok(resp("turn 2")),
        Ok(resp("turn 3")),
    ]);
    let result = s.last_ok().await;
    assert_eq!(result.unwrap().content.as_deref(), Some("turn 3"));
}

#[tokio::test]
async fn last_ok_skips_errors() {
    let s = AgentTurnsStream::from_items(vec![
        Ok(resp("turn 1")),
        Err(mock_err()),
        Ok(resp("turn 2")),
        Err(mock_err()),
    ]);
    let result = s.last_ok().await;
    assert_eq!(result.unwrap().content.as_deref(), Some("turn 2"));
}

#[tokio::test]
async fn last_ok_all_errors_returns_none() {
    let s = AgentTurnsStream::from_items(vec![Err(mock_err()), Err(mock_err())]);
    assert!(s.last_ok().await.is_none());
}

#[tokio::test]
async fn last_ok_empty_stream_returns_none() {
    let s = AgentTurnsStream::from_items(vec![]);
    assert!(s.last_ok().await.is_none());
}

#[tokio::test]
async fn last_ok_single_ok() {
    let s = AgentTurnsStream::from_items(vec![Ok(resp("only"))]);
    assert_eq!(s.last_ok().await.unwrap().content.as_deref(), Some("only"));
}

// ── last_content() ────────────────────────────────────────────────────────────

#[tokio::test]
async fn last_content_returns_last_text() {
    let s = AgentTurnsStream::from_items(vec![Ok(resp("first")), Ok(resp("second"))]);
    assert_eq!(s.last_content().await, "second");
}

#[tokio::test]
async fn last_content_empty_stream_returns_empty_string() {
    let s = AgentTurnsStream::from_items(vec![]);
    assert_eq!(s.last_content().await, "");
}

#[tokio::test]
async fn last_content_all_errors_returns_empty_string() {
    let s = AgentTurnsStream::from_items(vec![Err(mock_err()), Err(mock_err())]);
    assert_eq!(s.last_content().await, "");
}

#[tokio::test]
async fn last_content_none_content_returns_empty_string() {
    let s = AgentTurnsStream::from_items(vec![Ok(resp_no_content())]);
    assert_eq!(s.last_content().await, "");
}

#[tokio::test]
async fn last_content_skips_errors_takes_last_ok() {
    let s = AgentTurnsStream::from_items(vec![Ok(resp("good")), Err(mock_err())]);
    assert_eq!(s.last_content().await, "good");
}

// ── manual while-let driving ──────────────────────────────────────────────────

#[tokio::test]
async fn agent_turns_stream_can_be_driven_manually() {
    let mut s = AgentTurnsStream::from_items(vec![
        Ok(resp("a")),
        Err(mock_err()),
        Ok(resp("b")),
    ]);
    let mut results = vec![];
    while let Some(item) = s.next().await {
        results.push(item.is_ok());
    }
    assert_eq!(results, vec![true, false, true]);
}

#[tokio::test]
async fn agent_turns_stream_manual_last_pattern() {
    let mut s = AgentTurnsStream::from_items(vec![Ok(resp("first")), Ok(resp("last"))]);
    let mut last = None::<CompleteResponse>;
    while let Some(Ok(r)) = s.next().await {
        last = Some(r);
    }
    assert_eq!(last.unwrap().content.as_deref(), Some("last"));
}
