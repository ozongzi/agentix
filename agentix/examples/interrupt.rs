//! Example: queuing a follow-up message while the agent is still running a tool,
//! and aborting a turn in progress via AgentInput::Abort.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example interrupt

use agentix::{AgentEvent, AgentInput, ToolOutput, streaming_tool};
use async_stream::stream;
use futures::StreamExt;
use serde_json::json;
use std::io::Write;
use tokio::time::{Duration, sleep};

struct SlowCounter;

#[streaming_tool]
impl Tool for SlowCounter {
    /// Count from 1 to n with a 200 ms delay per step.
    /// n: how high to count
    fn count_to(&self, n: u32) -> impl futures::Stream<Item = ToolOutput> {
        stream! {
            for i in 1..=n {
                sleep(Duration::from_millis(200)).await;
                yield ToolOutput::Progress(format!("{i}/{n}"));
            }
            yield ToolOutput::Result(json!({ "final_count": n }));
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let token = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");

    // ── Scenario A: follow-up queued mid-tool via subscribe() ────────────────
    println!("=== Scenario A: follow-up message queued mid-tool ===\n");

    let mut agent = agentix::deepseek(&token)
        .system_prompt(
            "You are a helpful assistant. \
             When asked to count, use count_to. \
             Always acknowledge follow-up messages.",
        )
        .tool(SlowCounter);

    // Send first message, then subscribe to the raw stream (doesn't stop at Done).
    agent.send("Count to 5 using count_to.").await?;

    // Queue a follow-up 500 ms later via a cloned sender.
    let tx2 = agent.sender();
    tokio::spawn(async move {
        sleep(Duration::from_millis(500)).await;
        println!("\n[queuing follow-up]\n");
        tx2.send(AgentInput::User(vec!["Also tell me the square of that number.".into()]))
            .await.ok();
    });

    let mut rx = agent.subscribe();
    while let Some(event) = rx.next().await {
        match event {
            AgentEvent::Token(t) => { print!("{t}"); std::io::stdout().flush().ok(); }
            AgentEvent::ToolCall(tc) => println!("\n[calling {}]", tc.name),
            AgentEvent::ToolProgress { name, progress, .. } => eprintln!("  [tool {}] {}", name, progress),
            AgentEvent::ToolResult { name, result, .. } => println!("[{name}] -> {result}"),
            AgentEvent::Done => { println!("\n--- turn done ---"); break; }
            AgentEvent::Error(e) => { eprintln!("Error: {e}"); break; }
            _ => {}
        }
    }

    // ── Scenario B: abort ────────────────────────────────────────────────────
    println!("\n=== Scenario B: abort ===\n");

    let mut agent = agentix::deepseek(token)
        .system_prompt("You are a helpful assistant. When asked to count, use count_to.")
        .tool(SlowCounter);

    agent.send("Count to 10 using count_to.").await?;

    let tx3 = agent.sender();
    let mut rx = agent.subscribe();
    while let Some(event) = rx.next().await {
        match event {
            AgentEvent::Token(t) => { print!("{t}"); std::io::stdout().flush().ok(); }
            AgentEvent::ToolCall(tc) => println!("\n[calling {}]", tc.name),
            AgentEvent::ToolProgress { name, progress, .. } => {
                eprintln!("  [tool {}] {}", name, progress);
                if progress.contains("2/10") {
                    println!("\n[aborting turn]\n");
                    tx3.send(AgentInput::Abort).await.ok();
                }
            }
            AgentEvent::Done => { println!("\n--- turn aborted ---"); break; }
            _ => {}
        }
    }
    println!("Agent ready for next message after abort.");

    Ok(())
}
