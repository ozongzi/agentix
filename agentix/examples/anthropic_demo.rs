//! Example: chat using the Anthropic Messages API (Claude).
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-ant-... cargo run --example anthropic_demo

use agentix::{AgentEvent, AnthropicAgent};
use futures::StreamExt;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");

    let mut agent = AnthropicAgent::official(api_key, "claude-sonnet-4-5")
        .streaming()
        .with_system_prompt("You are a helpful assistant.");

    println!("Chatting with Claude via Anthropic API (claude-sonnet-4-5).");
    println!("Type a prompt and press Enter. Ctrl+C to exit.\n");

    let mut line = String::new();

    loop {
        print!("> ");
        io::stdout().flush()?;

        line.clear();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }

        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }

        let mut stream = agent.chat(prompt);

        while let Some(event) = stream.next().await {
            match event {
                Err(e) => {
                    eprintln!("\nError: {e}");
                    break;
                }
                Ok(AgentEvent::Token(text)) => {
                    print!("{text}");
                    io::stdout().flush().ok();
                }
                Ok(AgentEvent::ReasoningToken(text)) => {
                    // Claude 3.7+ "thinking" blocks are automatically handled.
                    print!("\n[Thinking] {text}");
                    io::stdout().flush().ok();
                }
                Ok(AgentEvent::ToolCall(c)) => {
                    if c.delta.is_empty() {
                        println!("\n[calling {}]", c.name);
                    } else {
                        print!("{}", c.delta);
                        io::stdout().flush().ok();
                    }
                }
                Ok(AgentEvent::ToolResult(r)) => {
                    println!("\n[result: {}]", r.result);
                }
            }
        }

        println!("\n");

        if let Some(a) = stream.into_agent() {
            agent = a;
        } else {
            break;
        }
    }

    Ok(())
}
