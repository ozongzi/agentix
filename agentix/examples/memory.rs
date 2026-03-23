//! Multi-turn conversation with persistent memory (SlidingWindow).
//!
//! Shows how the agent remembers context across turns — ask a follow-up
//! question that only makes sense given the previous answer.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example memory

use agentix::{Msg, SlidingWindow};
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");

    // SlidingWindow keeps the last 10 turns; InMemory (default) keeps all.
    let agent = agentix::deepseek(key)
        .system_prompt("You are a helpful assistant. Keep answers brief.")
        .memory(SlidingWindow::new(10))
        .max_tokens(256);

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

        let mut rx = agent.subscribe();
        agent.send(prompt).await;

        while let Ok(msg) = rx.recv().await {
            match msg {
                Msg::Token(t) => { print!("{t}"); std::io::stdout().flush().ok(); }
                Msg::Done     => break,
                Msg::Error(e) => { eprintln!("Error: {e}"); break; }
                _             => {}
            }
        }
        println!("\n");
    }

    Ok(())
}
