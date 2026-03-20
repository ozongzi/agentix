use agentix::{AgentEvent, DeepSeekAgent, tool};
use futures::StreamExt;
use reqwest::Client;
use serde_json::json;

struct WeatherTool {
    client: Client,
}

#[tool]
impl Tool for WeatherTool {
    /// Get current weather for a city.
    /// city: city name
    /// unit: temperature unit, e.g., "celsius" or "fahrenheit" (optional)
    async fn get_weather(&self, city: String, unit: Option<String>) -> Value {
        let _ = unit;
        let url = format!("https://wttr.in/{}?format=3", city);
        let text = match self.client.get(&url).send().await {
            Ok(response) => match response.text().await {
                Ok(body) => body,
                Err(e) => e.to_string(),
            },
            Err(e) => e.to_string(),
        };
        json!({ "city": city, "weather": text })
    }
}

#[tokio::main]
async fn main() {
    // Ensure DEEPSEEK_API_KEY is set in your environment before running this example.
    let token = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");

    // ── Non-streaming (default) ───────────────────────────────────────────────
    // The agent waits for the full response before yielding it.
    println!("=== Non-streaming ===");

    let agent = DeepSeekAgent::new(&token)
        .with_tool(WeatherTool {
            client: Client::new(),
        })
        .with_system_prompt("You are a helpful assistant.");

    let mut stream = agent.chat("Check the weather for Beijing and Shanghai");

    while let Some(event) = stream.next().await {
        match event {
            Err(e) => {
                eprintln!("Error: {e}");
                break;
            }
            Ok(AgentEvent::Token(text)) => println!("Assistant: {}", text),
            Ok(AgentEvent::ReasoningToken(text)) => println!("Reasoning: {}", text),
            Ok(AgentEvent::ToolCall(c)) => {
                if c.delta.is_empty() { println!("Tool call start: {} (id={})", c.name, c.id) }
                else { println!("Tool call {}({})", c.name, c.delta) }
            }
            Ok(AgentEvent::ToolResult(r)) => println!("-> {}", r.result),
        }
    }

    // ── Streaming ────────────────────────────────────────────────────────────
    // With `.streaming()` the agent uses SSE internally.
    // Text fragments are yielded one by one as they arrive; tool call results
    // still arrive as a discrete event after execution completes.
    println!("\n=== Streaming ===");

    let agent = DeepSeekAgent::new(&token)
        .streaming()
        .with_tool(WeatherTool {
            client: Client::new(),
        })
        .with_system_prompt("You are a helpful assistant.");

    let mut stream = agent.chat("Check the weather for Beijing and Shanghai");

    while let Some(event) = stream.next().await {
        match event {
            Err(e) => {
                eprintln!("Error: {e}");
                break;
            }
            // In streaming mode each Token is a single text fragment —
            // print without a newline to show them inline.
            Ok(AgentEvent::Token(text)) => print!("{}", text),
            Ok(AgentEvent::ReasoningToken(text)) => print!("{}", text),
            Ok(AgentEvent::ToolCall(c)) => {
                if c.delta.is_empty() { println!("\n[calling {}  id={}]", c.name, c.id) }
                else { print!("{}", c.delta) }
            }
            Ok(AgentEvent::ToolResult(r)) => println!("[result] {}", r.result),
        }
    }
    println!(); // final newline after streamed text
}
