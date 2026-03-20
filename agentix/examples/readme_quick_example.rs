use agentix::{AgentEvent, DeepSeekAgent, tool};
use futures::StreamExt;
use serde_json::json;

struct Echo;

#[tool]
impl Tool for Echo {
    /// Echo the input string back as JSON.
    async fn echo(&self, input: String) -> Value {
        json!({ "echo": input })
    }
}

#[tokio::main]
async fn main() {
    // Ensure DEEPSEEK_API_KEY is set in your environment before running this example.
    let token = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");

    let agent = DeepSeekAgent::new(&token).with_tool(Echo);

    // Use the agent to chat (non-streaming example).
    let mut stream = agent.chat("Please echo: hello");

    while let Some(event) = stream.next().await {
        match event {
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
            Ok(AgentEvent::Token(text)) => {
                println!("Assistant: {}", text);
            }
            Ok(AgentEvent::ReasoningToken(text)) => {
                println!("[reasoning] {}", text);
            }
            Ok(AgentEvent::ToolCall(c)) => {
                if c.delta.is_empty() { println!("[tool call start] {} (id={})", c.name, c.id); }
                else { println!("[tool call args] {}: {}", c.id, c.delta); }
            }
            Ok(AgentEvent::ToolResult(res)) => {
                println!("[tool result] {} -> {}", res.name, res.result);
            }
        }
    }
}
