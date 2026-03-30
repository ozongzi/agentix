use agentix::{LlmEvent, Request};
use futures::StreamExt;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // You can swap this to Provider::OpenAI, Provider::Anthropic, etc.
    let api_key = env::var("DEEPSEEK_API_KEY")
        .expect("DEEPSEEK_API_KEY must be set in your environment variables");

    let http = reqwest::Client::new();

    println!("Sending request to DeepSeek...");

    let mut stream = Request::deepseek(api_key)
        .system_prompt("You are a helpful and concise AI assistant.")
        .user("Write a short haiku about Rust programming.")
        .stream(&http)
        .await?;

    println!("\nResponse:");

    // Process the stream of events
    while let Some(event) = stream.next().await {
        match event {
            LlmEvent::Reasoning(r) => {
                // Print reasoning in cyan if the model supports it (e.g., DeepSeek R1)
                print!("\x1b[36m{r}\x1b[0m");
            }
            LlmEvent::Token(t) => {
                // Print standard output tokens
                print!("{t}");
            }
            LlmEvent::Done => {
                break;
            }
            LlmEvent::Error(e) => {
                eprintln!("\nError encountered: {e}");
            }
            _ => {
                // Ignore other events like ToolCall or Usage for this basic example
            }
        }
    }

    println!("\n");

    Ok(())
}
