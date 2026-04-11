//! Tests for `Agent` and `AgentEvent`.
//!
//! These tests exercise the `Agent` struct without making real LLM calls by
//! using a mock `Tool` and inspecting the `AgentEvent` stream produced by
//! `Agent::new()` / `Agent::from_arc()` / `Agent::max_iterations()`.

use agentix::tool_trait::Tool;
use agentix::{AgentEvent, ToolBundle};
use serde_json::json;
use std::sync::Arc;

// ── Helpers ───────────────────────────────────────────────────────────────────

// ── Mock tools ────────────────────────────────────────────────────────────────

struct EchoTool;

#[agentix::tool]
impl Tool for EchoTool {
    /// Echo the input back.
    /// input: text to echo
    async fn echo(&self, input: String) -> String {
        input
    }
}

struct FailTool;

#[agentix::tool]
impl Tool for FailTool {
    /// Always returns an error value.
    async fn fail(&self) -> serde_json::Value {
        json!({ "error": "deliberate failure" })
    }
}

// ── Construction tests ────────────────────────────────────────────────────────

#[test]
fn agent_fn_returns_boxstream() {
    // Just verify that agentix::agent() compiles and produces a BoxStream.
    // We don't drive it (would need a real LLM) — type-checking is enough.
    let _stream = agentix::agent(
        ToolBundle::default(),
        reqwest::Client::new(),
        agentix::Request::new(agentix::Provider::OpenAI, "sk-test"),
        vec![],
        None,
    );
}

#[test]
fn agent_fn_accepts_arc_tool() {
    let arc: Arc<dyn Tool> = Arc::new(EchoTool);
    let _stream = agentix::agent(
        arc,
        reqwest::Client::new(),
        agentix::Request::new(agentix::Provider::OpenAI, "sk-test"),
        vec![],
        None,
    );
}

#[test]
fn agent_tool_defs_exposed() {
    use agentix::tool_trait::Tool as _;
    let bundle = ToolBundle::default() + EchoTool + FailTool;
    let defs = bundle.raw_tools();
    let names: Vec<&str> = defs.iter().map(|d| d.function.name.as_str()).collect();
    assert!(names.contains(&"echo"), "expected 'echo' in tool defs");
    assert!(names.contains(&"fail"), "expected 'fail' in tool defs");
}

// ── AgentEvent enum tests ─────────────────────────────────────────────────────

#[test]
fn agent_event_token_clone() {
    let ev = AgentEvent::Token("hello".into());
    let ev2 = ev.clone();
    assert!(matches!(ev2, AgentEvent::Token(t) if t == "hello"));
}

#[test]
fn agent_event_debug() {
    let ev = AgentEvent::Error("oops".into());
    let s = format!("{ev:?}");
    assert!(s.contains("oops"));
}

#[test]
fn agent_event_reasoning() {
    let ev = AgentEvent::Reasoning("thinking...".into());
    assert!(matches!(ev, AgentEvent::Reasoning(t) if t == "thinking..."));
}

#[test]
fn agent_event_warning() {
    let ev = AgentEvent::Warning("truncated".into());
    assert!(matches!(ev, AgentEvent::Warning(_)));
}

#[test]
fn agent_event_tool_progress() {
    let ev = AgentEvent::ToolProgress {
        id: "id1".into(),
        name: "search".into(),
        progress: "50%".into(),
    };
    if let AgentEvent::ToolProgress { id, name, progress } = ev {
        assert_eq!(id, "id1");
        assert_eq!(name, "search");
        assert_eq!(progress, "50%");
    } else {
        panic!("wrong variant");
    }
}

#[test]
fn agent_event_tool_result() {
    let ev = AgentEvent::ToolResult {
        id: "c1".into(),
        name: "echo".into(),
        content: vec![agentix::request::Content::text("hello")],
    };
    if let AgentEvent::ToolResult { id, name, content } = ev {
        assert_eq!(id, "c1");
        assert_eq!(name, "echo");
        assert_eq!(content, vec![agentix::request::Content::text("hello")]);
    } else {
        panic!("wrong variant");
    }
}
