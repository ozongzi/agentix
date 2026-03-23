//! Quickstart: one-shot question with streaming output.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example readme_quick_example

use agentix::Msg;
use std::io::Write;

#[tokio::main]
async fn main() {
    let agent = agentix::deepseek(std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set"))
        .system_prompt("You are a helpful assistant.")
        .max_tokens(1024);

    let mut rx = agent.subscribe();
    agent.send("What is the capital of France?").await;

    while let Ok(msg) = rx.recv().await {
        match msg {
            Msg::Token(t) => { print!("{t}"); std::io::stdout().flush().ok(); }
            Msg::Done     => break,
            _             => {}
        }
    }
    println!();
}
