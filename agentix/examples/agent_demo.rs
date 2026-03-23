//! Agent with a tool: live weather lookup.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example agent_demo

use agentix::{AgentEvent, tool};
use futures::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::io::Write;

struct WeatherTool {
    client: Client,
}

#[tool]
impl agentix::Tool for WeatherTool {
    /// Get current weather for a city.
    /// city: city name
    async fn get_weather(&self, city: String) -> Result<Value, String> {
        let url = format!("https://wttr.in/{}?format=3", city);
        let resp = self.client.get(&url).send().await
            .map_err(|e| e.to_string())?;
        let text = resp.text().await
            .map_err(|e| e.to_string())?;
        Ok(json!({ "city": city, "weather": text }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let token = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");

    let mut agent = agentix::deepseek(token)
        .system_prompt("You are a helpful assistant.")
        .tool(WeatherTool { client: Client::new() });

    let mut stream = agent.chat("Check the weather for Beijing and Shanghai.").await?;
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::Token(t)     => { print!("{t}"); std::io::stdout().flush().ok(); }
            AgentEvent::Reasoning(r) => { print!("\x1b[2m{r}\x1b[0m"); std::io::stdout().flush().ok(); }
            AgentEvent::ToolCall(tc) => println!("\n[calling {}({})]", tc.name, tc.arguments),
            AgentEvent::ToolProgress { name, progress, .. } => eprintln!("  [tool {}] {}", name, progress),
            AgentEvent::ToolResult { name, result, .. } => println!("[{name}] -> {result}"),
            AgentEvent::Error(e) => { eprintln!("Error: {e}"); break; }
            _ => {}
        }
    }
    println!();
    Ok(())
}
