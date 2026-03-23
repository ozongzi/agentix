//! Agent with a tool: live weather lookup.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example agent_demo

use agentix::{AgentEvent, AgentInput, Node, tool};
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
async fn main() {
    let token = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");

    let agent = agentix::deepseek(token)
        .system_prompt("You are a helpful assistant.")
        .tool(WeatherTool { client: Client::new() }).await;

    // Create an input stream
    let input = futures::stream::iter(vec![
        AgentInput::User(vec!["Check the weather for Beijing and Shanghai.".into()])
    ]).boxed();

    // Run the agent and get the response stream
    let mut response = agent.run(input);

    while let Some(event) = response.next().await {
        match event {
            AgentEvent::Token(t)  => { print!("{t}"); std::io::stdout().flush().ok(); }
            AgentEvent::Reasoning(r) => { print!("\x1b[2m{r}\x1b[0m"); std::io::stdout().flush().ok(); }
            AgentEvent::ToolCallChunk(_) => {}
            AgentEvent::ToolCall(tc) => println!("\n[calling {}({})]", tc.name, tc.arguments),
            AgentEvent::ToolResult { name, result, .. } => println!("[{name}] → {result}"),
            AgentEvent::Done  => break,
            AgentEvent::Error(e) => { eprintln!("Error: {e}"); break; }
            _ => {}
        }
    }
    println!();
}
