//! Example: queuing a follow-up message while the agent is still running a tool,
//! and aborting a turn in progress with `agent.abort()`.
//!
//! Two scenarios are shown:
//!
//! **Scenario A — follow-up queued mid-tool**: the agent starts a slow tool;
//! a background task calls `agent.send()` after 500 ms.  The follow-up is
//! appended to the inbox and processed after the current tool round finishes.
//!
//! **Scenario B — abort**: `agent.abort()` cancels the current turn; the
//! agent emits `Msg::Done` and is ready for the next message immediately.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example interrupt

use agentix::{Msg, tool};
use serde_json::json;
use std::io::Write;
use tokio::time::{Duration, sleep};

struct SlowCounter;

#[tool]
impl agentix::Tool for SlowCounter {
    /// Count from 1 to n with a 200 ms delay per step.
    /// n: how high to count
    async fn count_to(&self, n: u32) -> serde_json::Value {
        for i in 1..=n {
            sleep(Duration::from_millis(200)).await;
            eprintln!("  [tool] {i}/{n}");
        }
        json!({ "final_count": n })
    }
}

async fn drain(rx: &mut tokio::sync::broadcast::Receiver<Msg>) {
    loop {
        match rx.recv().await {
            Ok(Msg::Token(t))     => { print!("{t}"); std::io::stdout().flush().ok(); }
            Ok(Msg::ToolCall { name, .. })     => println!("\n[calling {name}]"),
            Ok(Msg::ToolResult { name, result, .. }) => println!("[{name}] -> {result}"),
            Ok(Msg::Done)  => { println!(); break; }
            Ok(Msg::Error(e)) => { eprintln!("Error: {e}"); break; }
            Err(_) | Ok(_) => break,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let token = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");

    // ── Scenario A: follow-up queued while tool is running ────────────────────
    println!("=== Scenario A: follow-up message queued mid-tool ===\n");

    let agent = agentix::deepseek(&token)
        .system_prompt(
            "You are a helpful assistant. \
             When asked to count, use count_to. \
             Always acknowledge follow-up messages.",
        )
        .tool(SlowCounter);

    let mut rx = agent.subscribe();
    agent.send("Count to 5 using count_to.").await;

    // Queue a follow-up 500 ms later — while the tool is still running.
    let agent2 = agent.clone();
    tokio::spawn(async move {
        sleep(Duration::from_millis(500)).await;
        println!("\n[queuing follow-up]\n");
        agent2.send("Also tell me the square of that number.").await;
    });

    // First turn (count_to)
    drain(&mut rx).await;
    // Second turn (square — queued follow-up)
    drain(&mut rx).await;

    // ── Scenario B: abort ────────────────────────────────────────────────────
    println!("\n=== Scenario B: abort ===\n");

    let agent = agentix::deepseek(token)
        .system_prompt("You are a helpful assistant.")
        .tool(SlowCounter);

    let mut rx = agent.subscribe();
    agent.send("Count to 10 using count_to.").await;

    // Abort after the tool starts but before it finishes.
    let agent3 = agent.clone();
    tokio::spawn(async move {
        sleep(Duration::from_millis(300)).await;
        println!("\n[aborting turn]\n");
        agent3.abort().await;
    });

    drain(&mut rx).await;
    println!("Agent ready for next message after abort.");

    Ok(())
}
