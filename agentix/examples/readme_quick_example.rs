//! Quickstart: one-shot question with streaming output.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example readme_quick_example

use agentix::AgentEvent;
use futures::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = agentix::deepseek(std::env::var("DEEPSEEK_API_KEY")?)
        .system_prompt("You are a helpful assistant.")
        .max_tokens(1024);

    let mut stream = agent.chat("What is the capital of France?").await?;
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::Token(t) => { print!("{t}"); std::io::stdout().flush().ok(); }
            AgentEvent::Error(e) => { eprintln!("Error: {e}"); break; }
            _ => {}
        }
    }
    println!();
    Ok(())
}
