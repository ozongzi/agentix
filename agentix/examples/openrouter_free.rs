//! REPL: chat via OpenRouter (OpenAI-compatible endpoint).
//!
//! OpenRouter exposes an OpenAI-compatible API; we just override base_url and model.
//!
//! Run with:
//!   OPENROUTER_API_KEY=sk-or-... cargo run --example openrouter_free

use agentix::AgentEvent;
use futures::StreamExt;
use std::io::{self, Write};

const BASE_URL: &str = "https://openrouter.ai/api/v1";
const MODEL:    &str = "openrouter/free";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set");

    let mut agent = agentix::openai(api_key)
        .base_url(BASE_URL)
        .model(MODEL)
        .system_prompt("You are a helpful assistant.");

    println!("Chatting with {MODEL} via OpenRouter. Ctrl+C to exit.\n");

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
                AgentEvent::ToolResult { name, result, .. } => println!("[{name}] -> {result}"),
                AgentEvent::Error(e) => { eprintln!("\nError: {e}"); break; }
                _ => {}
            }
        }
        println!("\n");
    }
    Ok(())
}
