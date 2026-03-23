//! Agent with a tool: live weather lookup.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example agent_demo

use agentix::{Msg, tool};
use reqwest::Client;
use serde_json::json;
use std::io::Write;

struct WeatherTool {
    client: Client,
}

#[tool]
impl agentix::Tool for WeatherTool {
    /// Get current weather for a city.
    /// city: city name
    async fn get_weather(&self, city: String) -> serde_json::Value {
        let url = format!("https://wttr.in/{}?format=3", city);
        let text = self.client.get(&url).send().await
            .and_then(|r| futures::executor::block_on(r.text()))
            .unwrap_or_else(|e| e.to_string());
        json!({ "city": city, "weather": text })
    }
}

#[tokio::main]
async fn main() {
    let token = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");

    let agent = agentix::deepseek(token)
        .system_prompt("You are a helpful assistant.")
        .tool(WeatherTool { client: Client::new() });

    let mut rx = agent.subscribe();
    agent.send("Check the weather for Beijing and Shanghai.").await;

    while let Ok(msg) = rx.recv().await {
        match msg {
            Msg::TurnStart => println!("--- turn start ---"),
            Msg::Token(t)  => { print!("{t}"); std::io::stdout().flush().ok(); }
            Msg::Reasoning(r) => { print!("\x1b[2m{r}\x1b[0m"); std::io::stdout().flush().ok(); }
            Msg::ToolCall { name, args, .. } => println!("\n[calling {name}({args})]"),
            Msg::ToolResult { name, result, .. } => println!("[{name}] → {result}"),
            Msg::Done  => break,
            Msg::Error(e) => { eprintln!("Error: {e}"); break; }
            _ => {}
        }
    }
    println!();
}
