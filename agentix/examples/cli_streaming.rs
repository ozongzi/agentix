//! CLI streaming REPL with a shell tool.
//!
//! The agent maintains conversation history across turns.  A shell.run tool
//! lets the model execute commands and incorporate the output.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example cli_streaming

use agentix::{AgentEvent, tool};
use futures::StreamExt;
use serde_json::{Value, json};
use std::io::{self, Write};
use std::process::Stdio;
use tokio::process::Command;

struct ShellTool;

#[tool]
impl agentix::Tool for ShellTool {
    /// Run a shell command and return stdout/stderr.
    /// command: the shell command to execute
    async fn run(&self, command: String) -> Result<Value, String> {
        let output = Command::new("sh")
            .arg("-c")
            .arg(&command)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| e.to_string())?;

        Ok(json!({
            "status": output.status.code(),
            "stdout": String::from_utf8_lossy(&output.stdout),
            "stderr": String::from_utf8_lossy(&output.stderr),
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let token = std::env::var("DEEPSEEK_API_KEY")?;

    let mut agent = agentix::deepseek(token)
        .system_prompt(
            "You may call shell.run(command) to execute shell commands. \
             Avoid destructive operations.",
        )
        .tool(ShellTool);

    println!("DeepSeek REPL (with shell tool). Ctrl+C to exit.\n");

    let mut line = String::new();
    loop {
        print!("> ");
        io::stdout().flush()?;
        line.clear();
        if io::stdin().read_line(&mut line)? == 0 { break; }
        let prompt = line.trim();
        if prompt.is_empty() { continue; }

        let mut stream = agent.chat(prompt).await?;
        while let Some(ev) = stream.next().await {
            match ev {
                AgentEvent::Token(t) => { print!("{t}"); io::stdout().flush().ok(); }
                AgentEvent::Reasoning(r) => { print!("\x1b[2m{r}\x1b[0m"); io::stdout().flush().ok(); }
                AgentEvent::ToolCallChunk(tc) => {
                    if tc.delta.is_empty() { print!("\n[calling {}... ", tc.name); }
                    else { print!("{}", tc.delta); }
                    io::stdout().flush().ok();
                }
                AgentEvent::ToolCall(_) => { println!("]"); }
                AgentEvent::ToolResult { name, result, .. } => println!("[{name}] -> {result}"),
                AgentEvent::Error(e) => { eprintln!("\nError: {e}"); break; }
                _ => {}
            }
        }
        println!("\n");
    }
    Ok(())
}
