//! Example 10: Claude Code (Max OAuth) via `Provider::ClaudeCode`
//!
//! Drives `claude -p` as a single-turn LLM through the standard `agent()`
//! loop. Tool calls dispatch locally through agentix's `Tool` trait — the
//! loopback MCP server only surfaces tool schemas. Uses the user's Claude
//! Max subscription; no API key needed.
//!
//! Run with:
//!   cargo run --example 10_claude_code --features claude-code

use agentix::{AgentEvent, Message, Request, UserContent, agent, tool};
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

    let http = reqwest::Client::new();
    let base = Request::claude_code()
        .model("sonnet")
        .system_prompt(
            "You are a concise math assistant. You MUST call the add/multiply \
             tools for every arithmetic operation — never compute in your head.",
        );
    let history = vec![Message::User(vec![UserContent::Text {
        text: "What is (123 + 456) * 789? Use your tools.".into(),
    }])];

    println!("Q: (123 + 456) * 789\n");

    let mut stream = agent(Calculator, http, base, history, None);
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
            AgentEvent::Usage(u) => eprintln!("\n[tokens: {}]", u.total_tokens),
            AgentEvent::Done(total) => {
                eprintln!("\n[total tokens: {}]", total.total_tokens);
                break;
            }
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
