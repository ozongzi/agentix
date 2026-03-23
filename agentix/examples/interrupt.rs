//! Example: queuing a follow-up message while the agent is still running a tool,
//! and aborting a turn in progress via AgentInput::Abort.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example interrupt

use agentix::{AgentEvent, AgentInput, Node, tool};
use serde_json::{json, Value};
use std::io::Write;
use tokio::time::{Duration, sleep};
use tokio::sync::mpsc;
use futures::StreamExt;

struct SlowCounter;

#[tool]
impl agentix::Tool for SlowCounter {
    /// Count from 1 to n with a 200 ms delay per step.
    /// n: how high to count
    async fn count_to(&self, n: u32) -> Result<Value, String> {
        for i in 1..=n {
            sleep(Duration::from_millis(200)).await;
            eprintln!("  [tool] {i}/{n}");
        }
        Ok(json!({ "final_count": n }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let token = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");

    // ── Scenario A: follow-up message queued mid-tool ────────────────────
    println!("=== Scenario A: follow-up message queued mid-tool ===\n");

    let agent = agentix::deepseek(&token)
        .system_prompt(
            "You are a helpful assistant. \
             When asked to count, use count_to. \
             Always acknowledge follow-up messages.",
        )
        .tool(SlowCounter).await;

    let (tx, rx) = mpsc::channel(64);
    let mut response = agent.run(tokio_stream::wrappers::ReceiverStream::new(rx).boxed());

    tx.send(AgentInput::User(vec!["Count to 5 using count_to.".into()])).await?;

    // Queue a follow-up 500 ms later — while the tool is still running.
    let tx2 = tx.clone();
    tokio::spawn(async move {
        sleep(Duration::from_millis(500)).await;
        println!("\n[queuing follow-up]\n");
        tx2.send(AgentInput::User(vec!["Also tell me the square of that number.".into()])).await.ok();
    });

    // Handle all events
    let mut turns_done = 0;
    while let Some(event) = response.next().await {
        match event {
            AgentEvent::Token(t)  => { print!("{t}"); std::io::stdout().flush().ok(); }
            AgentEvent::ToolCall(tc) => println!("\n[calling {}]", tc.name),
            AgentEvent::ToolResult { name, result, .. } => println!("[{name}] -> {result}"),
            AgentEvent::Done  => {
                println!("\n--- turn done ---");
                turns_done += 1;
                if turns_done >= 2 { break; }
            }
            AgentEvent::Error(e) => { eprintln!("Error: {e}"); break; }
            _ => {}
        }
    }

    // ── Scenario B: abort ────────────────────────────────────────────────────
    println!("\n=== Scenario B: abort ===\n");

    let agent = agentix::deepseek(token)
        .system_prompt("You are a helpful assistant.")
        .tool(SlowCounter).await;

    let (tx, rx) = mpsc::channel(64);
    let mut response = agent.run(tokio_stream::wrappers::ReceiverStream::new(rx).boxed());

    tx.send(AgentInput::User(vec!["Count to 10 using count_to.".into()])).await?;

    // Abort after the tool starts but before it finishes.
    let tx3 = tx.clone();
    tokio::spawn(async move {
        sleep(Duration::from_millis(300)).await;
        println!("\n[aborting turn]\n");
        tx3.send(AgentInput::Abort).await.ok();
    });

    while let Some(event) = response.next().await {
        match event {
            AgentEvent::Token(t)  => { print!("{t}"); std::io::stdout().flush().ok(); }
            AgentEvent::ToolCall(tc) => println!("\n[calling {}]", tc.name),
            AgentEvent::Done  => {
                println!("\n--- turn aborted ---");
                break;
            }
            _ => {}
        }
    }
    println!("Agent ready for next message after abort.");

    Ok(())
}
