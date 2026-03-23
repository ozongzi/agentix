//! REPL: chat with Claude (Anthropic).
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-ant-... cargo run --example anthropic_demo

use agentix::AgentEvent;
use futures::StreamExt;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");

    let mut agent = agentix::anthropic(api_key)
        .model("claude-sonnet-4-5")
        .system_prompt("You are a helpful assistant.");

    println!("Chatting with Claude (claude-sonnet-4-5). Ctrl+C to exit.\n");

    let mut line = String::new();
    loop {
        print!("> ");
        io::stdout().flush()?;
        line.clear();
        if io::stdin().read_line(&mut line)? == 0 { break; }
        let prompt = line.trim();
        if prompt.is_empty() { continue; }

        let mut stream = agent.chat(prompt).await?;
        while let Some(ev) = stream.next().await {
            match ev {
                AgentEvent::Token(t)     => { print!("{t}"); io::stdout().flush().ok(); }
                AgentEvent::Reasoning(r) => { print!("\x1b[2m{r}\x1b[0m"); io::stdout().flush().ok(); }
                AgentEvent::ToolCall(tc) => println!("\n[calling {}]", tc.name),
                AgentEvent::ToolResult { result, .. } => println!("[result] {result}"),
                AgentEvent::Error(e) => { eprintln!("\nError: {e}"); break; }
                _ => {}
            }
        }
        println!("\n");
    }
    Ok(())
}
