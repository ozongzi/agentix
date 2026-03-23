//! REPL: chat with Claude (Anthropic).
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-ant-... cargo run --example anthropic_demo

use agentix::Msg;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");

    let agent = agentix::anthropic(api_key)
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

        let mut rx = agent.subscribe();
        agent.send(prompt).await;

        loop {
            match rx.recv().await {
                Ok(Msg::Token(t))     => { print!("{t}"); io::stdout().flush().ok(); }
                Ok(Msg::Reasoning(r)) => { print!("\x1b[2m{r}\x1b[0m"); io::stdout().flush().ok(); }
                Ok(Msg::ToolCall { name, .. })       => println!("\n[calling {name}]"),
                Ok(Msg::ToolResult { result, .. })   => println!("[result] {result}"),
                Ok(Msg::Done)  => break,
                Ok(Msg::Error(e)) => { eprintln!("\nError: {e}"); break; }
                Err(_) | Ok(_) => break,
            }
        }
        println!("\n");
    }
    Ok(())
}
