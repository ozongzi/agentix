//! Example 07: Agentic loop with tools
//!
//! `agentix::agent()` drives the full LLM ↔ tool-call loop automatically.
//! It keeps calling the LLM until there are no more tool calls, yielding
//! typed `AgentEvent`s as it goes.
//!
//! Run with:
//!   OPENAI_API_KEY=sk-... cargo run --example 07_agent

use agentix::{AgentEvent, Message, Provider, Request, ToolBundle, UserContent, tool};
use futures::StreamExt;
use std::env;

// ── Tools ─────────────────────────────────────────────────────────────────────

struct Calculator;

#[tool]
impl agentix::Tool for Calculator {
    /// Add two numbers.
    /// a: first number
    /// b: second number
    async fn add(&self, a: f64, b: f64) -> f64 {
        a + b
    }

    /// Multiply two numbers.
    /// a: first number
    /// b: second number
    async fn multiply(&self, a: f64, b: f64) -> f64 {
        a * b
    }

    /// Divide a by b. Returns an error if b is zero.
    /// a: dividend
    /// b: divisor
    async fn divide(&self, a: f64, b: f64) -> Result<f64, String> {
        if b == 0.0 {
            Err("division by zero".into())
        } else {
            Ok(a / b)
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (provider, api_key) = if let Ok(k) = env::var("DEEPSEEK_API_KEY") {
        (Provider::DeepSeek, k)
    } else if let Ok(k) = env::var("OPENAI_API_KEY") {
        (Provider::OpenAI, k)
    } else {
        panic!("Set DEEPSEEK_API_KEY or OPENAI_API_KEY");
    };

    let http = reqwest::Client::new();
    let tools = ToolBundle::default() + Calculator;

    let request = Request::new(provider, api_key)
        .system_prompt("You are a math assistant. Use your tools to compute exact results.");

    let history = vec![
        Message::User(vec![UserContent::Text(
            "What is (123 + 456) * 789 / 3?".into(),
        )]),
    ];

    println!("Question: (123 + 456) * 789 / 3\n");

    let mut stream = agentix::agent(tools, 25_000, http, request, history);

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::Token(t) => print!("{t}"),
            AgentEvent::Reasoning(t) => print!("\x1b[2m{t}\x1b[0m"),
            AgentEvent::ToolCallStart(tc) => {
                println!("\n→ {}({})", tc.name, tc.arguments);
            }
            AgentEvent::ToolProgress { name, progress, .. } => {
                println!("  [{name}] {progress}");
            }
            AgentEvent::ToolResult { name, content, .. } => {
                println!("← [{name}] = {content}");
            }
            AgentEvent::Usage(u) => {
                eprintln!("\n[tokens: {}]", u.total_tokens);
            }
            AgentEvent::Warning(w) => eprintln!("\n[warn] {w}"),
            AgentEvent::Error(e) => {
                eprintln!("\n[error] {e}");
                break;
            }
            AgentEvent::ToolCallChunk(_) => {}
        }
    }

    println!();
    Ok(())
}
