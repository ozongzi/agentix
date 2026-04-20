use agentix::{LlmEvent, Request, Tool, tool};
use futures::StreamExt;
use std::env;

// ── Style 1: standalone function ─────────────────────────────────────────────
// #[tool] on a free async fn: generates a unit struct + Tool impl automatically.
// Use this for quick, one-off tools.

/// Add two numbers together.
/// a: first number
/// b: second number
#[agentix::tool]
async fn add(a: i64, b: i64) -> i64 {
    a + b
}

// ── Style 2: impl block ───────────────────────────────────────────────────────
// #[tool] on an impl block: multiple tools share one struct.
// Use this when tools share state (e.g. a DB connection, API client).

struct Calculator;

#[tool]
impl agentix::Tool for Calculator {
    /// Multiply two numbers.
    /// a: first number
    /// b: second number
    async fn multiply(&self, a: i64, b: i64) -> i64 {
        a * b
    }

    /// Divide a by b. Returns an error if b is zero.
    /// a: dividend
    /// b: divisor
    async fn divide(&self, a: f64, b: f64) -> Result<f64, String> {
        if b == 0.0 {
            Err("division by zero".into())
        } else {
            Ok(a / b)
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tools work best with OpenAI or Anthropic (though supported across providers)
    let api_key = env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY must be set in your environment variables");

    let http = reqwest::Client::new();

    // Combine standalone fns and impl-block structs with the + operator.
    // Both styles produce the same ToolBundle.
    let bundle = add + Calculator;

    println!("Sending request to OpenAI with calculator tools...");

    let mut stream = Request::openai(api_key)
        .model("gpt-4o")
        .system_prompt("You are a math assistant. You MUST use your tools to perform calculations.")
        .user("What is 1234 multiplied by 5678, then divided by 3?")
        .tools(bundle.raw_tools())
        .stream(&http)
        .await?;

    println!("\nResponse stream:");

    while let Some(event) = stream.next().await {
        match event {
            LlmEvent::Token(t) => {
                // Standard text output from the model
                print!("{t}");
            }
            LlmEvent::ToolCall(tc) => {
                // The model decided to call a tool
                println!("\n\n[Model requested tool call]");
                println!("Tool Name: {}", tc.name);
                println!("Arguments: {}", tc.arguments);

                // In a fully autonomous loop, you would now:
                // 1. Execute the tool using the bundle
                // 2. Create a Message::tool(...) with the result
                // 3. Append it to your conversation history
                // 4. Send a new request to the LLM
            }
            LlmEvent::Done => {
                break;
            }
            LlmEvent::Error(e) => {
                eprintln!("\nError: {e}");
            }
            _ => {}
        }
    }

    println!("\n");

    Ok(())
}
