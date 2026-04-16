//! Example 10: `claude -p` as the agentic loop (Claude Max OAuth)
//!
//! Drives the `claude` CLI in headless mode so tool calls dispatch back to
//! in-process agentix [`Tool`]s via a loopback MCP server. Uses the user's
//! Claude Max subscription — no API key needed.
//!
//! Run with:
//!   cargo run --example 10_claude_code --features claude-code

use agentix::{
    AgentEvent, ClaudeCodeConfig, Message, ToolBundle, UserContent, agent_claude_code, tool,
};
use futures::StreamExt;

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
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let tools = ToolBundle::default() + Calculator;

    let history = vec![Message::User(vec![UserContent::Text {
        text: "What is (123 + 456) * 789? Use your tools.".into(),
    }])];

    println!("Question: (123 + 456) * 789 — use tools\n");

    let mut stream = agent_claude_code(
        tools,
        "You are a concise math assistant. Always use the provided tools for arithmetic; never compute in your head.",
        ClaudeCodeConfig::new(),
        history,
    );

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::Token(t) => print!("{t}"),
            AgentEvent::Reasoning(t) => print!("\x1b[2m{t}\x1b[0m"),
            AgentEvent::ToolCallStart(tc) => {
                println!("\n→ {}({})", tc.name, tc.arguments);
            }
            AgentEvent::ToolResult { name, ref content, .. } => {
                let text = content
                    .iter()
                    .filter_map(|p| {
                        if let agentix::Content::Text { text } = p {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                println!("← [{name}] = {text}");
            }
            AgentEvent::Usage(u) => {
                eprintln!("\n[tokens: {}]", u.total_tokens);
            }
            AgentEvent::Done(total) => {
                eprintln!("\n[total tokens: {}]", total.total_tokens);
            }
            AgentEvent::Warning(w) => eprintln!("\n[warn] {w}"),
            AgentEvent::Error(e) => {
                eprintln!("\n[error] {e}");
                break;
            }
            _ => {}
        }
    }

    println!();
    Ok(())
}
