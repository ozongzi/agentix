//! Quickstart: one-shot question with streaming output.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example readme_quick_example

use agentix::{AgentInput, AgentEvent, Node};
use futures::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() {
    let agent = agentix::deepseek(std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set"))
        .system_prompt("You are a helpful assistant.")
        .max_tokens(1024);

    // Create an input stream
    let input = futures::stream::iter(vec![
        AgentInput::User(vec!["What is the capital of France?".into()])
    ]).boxed();

    // Run the agent and get the response stream
    let mut response = agent.run(input);

    while let Some(event) = response.next().await {
        match event {
            AgentEvent::Token(t) => { print!("{t}"); std::io::stdout().flush().ok(); }
            AgentEvent::Done     => break,
            _                    => {}
        }
    }
    println!();
}
