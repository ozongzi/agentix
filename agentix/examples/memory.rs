//! Multi-turn conversation with persistent memory (SlidingWindow).
//!
//! Shows how the agent remembers context across turns — ask a follow-up
//! question that only makes sense given the previous answer.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example memory

use agentix::{AgentEvent, SlidingWindow};
use futures::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");

    let mut agent = agentix::deepseek(key)
        .system_prompt("You are a helpful assistant. Keep answers brief.")
        .memory(SlidingWindow::new(10))
        .max_tokens(256);

    let turns = [
        "What is the largest planet in our solar system?",
        "How many Earths could fit inside it?",
        "What is its most famous feature?",
        "Name one of its moons.",
        "Is that moon larger than our Moon?",
    ];

    for (i, prompt) in turns.iter().enumerate() {
        println!("Turn {}: {prompt}", i + 1);
        print!("Agent: ");

        let mut stream = agent.chat(*prompt).await?;
        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::Token(t) => { print!("{t}"); std::io::stdout().flush().ok(); }
                AgentEvent::Error(e) => { eprintln!("Error: {e}"); break; }
                _ => {}
            }
        }
        println!("\n");
    }

    Ok(())
}
