//! CLI streaming REPL with a shell tool.
//!
//! The agent maintains conversation history across turns.  A shell.run tool
//! lets the model execute commands and incorporate the output.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example cli_streaming

use agentix::{Msg, tool};
use serde_json::json;
use std::io::{self, Write};
use std::process::Stdio;
use tokio::process::Command;

struct ShellTool;

#[tool]
impl agentix::Tool for ShellTool {
    /// Run a shell command and return stdout/stderr.
    /// command: the shell command to execute
    async fn run(&self, command: String) -> serde_json::Value {
        let output = Command::new("sh")
            .arg("-c")
            .arg(&command)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;
        match output {
            Ok(out) => json!({
                "status": out.status.code(),
                "stdout": String::from_utf8_lossy(&out.stdout),
                "stderr": String::from_utf8_lossy(&out.stderr),
            }),
            Err(e) => json!({ "error": e.to_string() }),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let token = std::env::var("DEEPSEEK_API_KEY")?;

    // The Agent is an Arc-based handle — shared, cloneable, persistent across turns.
    let agent = agentix::deepseek(token)
        .tool(ShellTool)
        .system_prompt(
            "You may call shell.run(command) to execute shell commands. \
             Avoid destructive operations.",
        );

    println!("DeepSeek REPL (with shell tool). Ctrl+C to exit.\n");

    let mut line = String::new();
    loop {
        print!("> ");
        io::stdout().flush()?;
        line.clear();
        if io::stdin().read_line(&mut line)? == 0 { break; }
        let prompt = line.trim();
        if prompt.is_empty() { continue; }

        let mut rx = agent.subscribe();
        agent.send(prompt).await;

        loop {
            match rx.recv().await {
                Ok(Msg::Token(t))     => { print!("{t}"); io::stdout().flush().ok(); }
                Ok(Msg::Reasoning(r)) => { print!("\x1b[2m{r}\x1b[0m"); io::stdout().flush().ok(); }
                Ok(Msg::ToolCall { name, args, .. }) => println!("\n[calling {name}({args})]"),
                Ok(Msg::ToolResult { name, result, .. }) => println!("[{name}] -> {result}"),
                Ok(Msg::Done)  => break,
                Ok(Msg::Error(e)) => { eprintln!("\nError: {e}"); break; }
                Err(_) | Ok(_) => break,
            }
        }
        println!("\n");
    }
    Ok(())
}
