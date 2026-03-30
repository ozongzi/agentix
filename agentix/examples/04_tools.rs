use agentix::{LlmEvent, Request, tool, Tool};
use futures::StreamExt;
use std::env;

// Define a struct that will hold our tools
struct Calculator;

// The #[tool] macro automatically generates the necessary metadata (JSON schemas)
// and dispatch code for the LLM to use these functions.
#[tool]
impl agentix::Tool for Calculator {
    /// Add two numbers together.
    /// a: first number
    /// b: second number
    async fn add(&self, a: i64, b: i64) -> i64 {
        a + b
    }

    /// Multiply two numbers.
    /// a: first number
    /// b: second number
    async fn multiply(&self, a: i64, b: i64) -> i64 {
        a * b
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tools work best with OpenAI or Anthropic (though supported across providers)
    let api_key = env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY must be set in your environment variables");

    let http = reqwest::Client::new();

    // A ToolBundle groups multiple tools together (including MCP tools)
    let mut bundle = agentix::ToolBundle::new();
    bundle.push(Calculator);

    println!("Sending request to OpenAI with calculator tools...");

    // Create the request and attach the tool definitions
    let mut stream = Request::openai(api_key)
        .model("gpt-4o")
        .system_prompt("You are a math assistant. You MUST use your tools to perform calculations.")
        .user("What is 1234 multiplied by 5678?")
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
