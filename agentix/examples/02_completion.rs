use agentix::Request;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // We'll use OpenAI for this example, but any provider works
    let api_key = env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY must be set in your environment variables");

    let http = reqwest::Client::new();

    println!("Sending non-streaming completion request to OpenAI...");

    // The `complete` method waits for the entire response to be generated
    // before returning the final text as a String.
    let response = Request::openai(api_key)
        .model("gpt-4o-mini") // Optional: override the default model
        .system_prompt("You are a helpful and concise AI assistant.")
        .user("Explain the theory of relativity in one short sentence.")
        .complete(&http)
        .await?;

    println!("\nResponse:\n{}", response.content.unwrap_or_default());

    Ok(())
}
