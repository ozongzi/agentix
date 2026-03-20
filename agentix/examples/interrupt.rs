//! Example: injecting user messages mid-loop via `with_interrupt_channel`.
//!
//! This example simulates a realistic scenario where the user sends a follow-up
//! message *while* the agent is still executing tools.  The injected message is
//! picked up automatically after the current tool-execution round finishes and
//! is included in the conversation history before the next API turn — so the
//! model naturally incorporates it into its next reply.
//!
//! What happens in this example:
//!
//! 1. The agent starts a multi-step task that intentionally calls a slow tool.
//! 2. A background task waits 500 ms (simulating the user typing) then sends a
//!    follow-up message through the interrupt channel.
//! 3. Once the tool round finishes, the agent sees the injected message and
//!    incorporates it into its next reply without any special handling on our end.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example interrupt

use std::io::Write;

use agentix::{AgentEvent, DeepSeekAgent, tool};
use futures::StreamExt;
use serde_json::json;
use tokio::time::{Duration, sleep};

// ── Tool definition ───────────────────────────────────────────────────────────

struct SlowCounter;

#[tool]
impl Tool for SlowCounter {
    /// Count from 1 to n, sleeping briefly between each step, and return the
    /// final count. Simulates a slow background task.
    /// n: how high to count
    async fn count_to(&self, n: u32) -> Value {
        for i in 1..=n {
            sleep(Duration::from_millis(200)).await;
            eprintln!("  [tool] counting… {i}/{n}");
        }
        json!({ "final_count": n, "message": format!("counted to {n}") })
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let token = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");

    // Build the agent and get the sender half of the interrupt channel.
    let agent = DeepSeekAgent::new(&token)
        .streaming()
        .with_system_prompt(
            "You are a helpful assistant. \
             When the user asks you to count, use the count_to tool. \
             Always acknowledge any follow-up messages from the user.",
        )
        .with_tool(SlowCounter);

    // Spawn a task that injects a follow-up message after a short delay.
    // This fires while the tool is still running (each step takes 200 ms,
    // the tool counts to 5 → ~1 s total; we inject at 500 ms).
    let tx_clone = agent.interrupt_sender();
    tokio::spawn(async move {
        sleep(Duration::from_millis(500)).await;
        println!("\n[user injects] \"Actually, please also tell me the square of that number.\"\n");
        tx_clone
            .send("Actually, please also tell me the square of that number.".into())
            .expect("channel closed unexpectedly");
    });

    println!("Asking the agent to count to 5 (tool takes ~1 s)…\n");

    let mut stream = agent.chat("Please count to 5 using the count_to tool.");

    while let Some(event) = stream.next().await {
        match event {
            Err(e) => {
                eprintln!("\nError: {e}");
                break;
            }

            Ok(AgentEvent::Token(fragment)) => {
                print!("{fragment}");
                std::io::stdout().flush().ok();
            }

            Ok(AgentEvent::ToolCall(c)) => {
                if c.delta.is_empty() {
                    println!("\n[calling {}]", c.name);
                }
            }

            Ok(AgentEvent::ToolResult(r)) => {
                println!("\n[tool result] {} -> {}", r.name, r.result);
                println!("\n(injected message will be picked up before the next API turn)\n");
            }

            Ok(_) => todo!(),
        }
    }

    println!("\n\n--- conversation complete ---");

    // The agent can be recovered and reused for further turns.
    if let Some(recovered) = stream.into_agent() {
        let history = recovered.history();
        println!("\nFinal history ({} messages):", history.len());
        for msg in history {
            use agentix::Message;
            let (role, preview) = match msg {
                Message::User(parts) => {
                    let text = parts.iter().filter_map(|p| {
                        if let agentix::request::UserContent::Text(t) = p { Some(t.as_str()) } else { None }
                    }).collect::<Vec<_>>().join(" ");
                    ("user", text.chars().take(80).collect::<String>())
                }
                Message::Assistant { content, .. } => {
                    let preview = content.as_deref().unwrap_or("<no content>")
                        .chars().take(80).collect::<String>();
                    ("assistant", preview)
                }
                Message::ToolResult { content, .. } => {
                    ("tool", content.chars().take(80).collect::<String>())
                }
            };
            println!("  [{role}] {preview}");
        }
    }

    Ok(())
}
