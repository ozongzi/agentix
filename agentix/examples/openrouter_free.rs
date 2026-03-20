//! Example: chat via OpenRouter using the free routing model.
//!
//! `openrouter/free` automatically routes to an available free-tier model.
//! OpenRouter exposes an OpenAI-compatible API, so all we need is:
//!   - a different base URL
//!   - a different model string
//!
//! Run with:
//!   OPENROUTER_API_KEY=sk-or-... cargo run --example openrouter_free

use agentix::{AgentEvent, OpenAIAgent};
use futures::StreamExt;
use std::io::{self, Write};

const BASE_URL: &str = "https://openrouter.ai/api/v1";
const MODEL: &str = "openrouter/free";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set");

    let mut agent = OpenAIAgent::new(&api_key, BASE_URL, MODEL)
        .streaming()
        .with_system_prompt("You are a helpful assistant.");

    println!("Chatting with {MODEL} via OpenRouter.");
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
                    print!("{text}");
                    io::stdout().flush().ok();
                }
                Ok(AgentEvent::ToolCall(c)) => {
                    if c.delta.is_empty() {
                        println!("\n[calling {}  id={}]", c.name, c.id);
                    } else {
                        print!("{}", c.delta);
                        io::stdout().flush().ok();
                    }
                }
                Ok(AgentEvent::ToolResult(r)) => {
                    println!("\n[tool result: {} -> {}]", r.name, r.result);
                }
            }
        }

        println!("\n");

        match stream.into_agent() {
            Some(a) => agent = a,
            None => {
                eprintln!("Agent was not returned from stream; exiting.");
                break;
            }
        }
    }

    Ok(())
}
