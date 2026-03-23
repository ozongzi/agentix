//! REPL: chat with Gemini (Google).
//!
//! Run with:
//!   GEMINI_API_KEY=AIza... cargo run --example gemini_demo

use agentix::{AgentEvent, AgentInput, Node};
use futures::StreamExt;
use std::io::{self, Write};
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let agent = agentix::gemini(api_key)
        .model("gemini-2.0-flash")
        .system_prompt("You are a helpful assistant.");

    println!("Chatting with Gemini (gemini-2.0-flash). Ctrl+C to exit.\n");

    let (tx, rx) = mpsc::channel(64);
    let mut response = agent.run(tokio_stream::wrappers::ReceiverStream::new(rx).boxed());

    let mut line = String::new();
    loop {
        print!("> ");
        io::stdout().flush()?;
        line.clear();
        if io::stdin().read_line(&mut line)? == 0 { break; }
        let prompt = line.trim();
        if prompt.is_empty() { continue; }

        tx.send(AgentInput::User(vec![prompt.into()])).await?;

        loop {
            match response.next().await {
                Some(AgentEvent::Token(t))   => { print!("{t}"); io::stdout().flush().ok(); }
                Some(AgentEvent::ToolCall(tc))     => println!("\n[calling {}]", tc.name),
                Some(AgentEvent::ToolResult { result, .. }) => println!("[result] {result}"),
                Some(AgentEvent::Done)  => break,
                Some(AgentEvent::Error(e)) => { eprintln!("\nError: {e}"); break; }
                None => break,
                _ => {}
            }
        }
        println!("\n");
    }
    Ok(())
}
