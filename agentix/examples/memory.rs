//! Multi-turn conversation with persistent memory (SlidingWindow).
//!
//! Shows how the agent remembers context across turns — ask a follow-up
//! question that only makes sense given the previous answer.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example memory

use agentix::{AgentEvent, AgentInput, Node, SlidingWindow};
use futures::StreamExt;
use std::io::Write;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");

    // SlidingWindow keeps the last 10 turns; InMemory (default) keeps all.
    let agent = agentix::deepseek(key)
        .system_prompt("You are a helpful assistant. Keep answers brief.")
        .memory(SlidingWindow::new(10)).await
        .max_tokens(256);

    let (tx, rx) = mpsc::channel(64);
    let mut response = agent.run(tokio_stream::wrappers::ReceiverStream::new(rx).boxed());

    let turns = [
        "What is the largest planet in our solar system?",
        "How many Earths could fit inside it?",            // references previous answer
        "What is its most famous feature?",               // still references Jupiter
        "Name one of its moons.",
        "Is that moon larger than our Moon?",             // references previous answer
    ];

    for (i, prompt) in turns.iter().enumerate() {
        println!("Turn {}: {prompt}", i + 1);
        print!("Agent: ");

        tx.send(AgentInput::User(vec![prompt.to_string().into()])).await?;

        while let Some(event) = response.next().await {
            match event {
                AgentEvent::Token(t) => { print!("{t}"); std::io::stdout().flush().ok(); }
                AgentEvent::Done     => break,
                AgentEvent::Error(e) => { eprintln!("Error: {e}"); break; }
                _             => {}
            }
        }
        println!("\n");
    }

    Ok(())
}
