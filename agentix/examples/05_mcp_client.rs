#[cfg(feature = "mcp")]
mod mcp_example {
    use agentix::{LlmEvent, McpTool, Provider, Request, Tool, ToolBundle};
    use futures::StreamExt;
    use std::env;
    use std::time::Duration;

    pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
        // We'll use Anthropic for this example, as Claude is excellent with tools.
        let api_key = env::var("ANTHROPIC_API_KEY")
            .expect("ANTHROPIC_API_KEY must be set in your environment variables");

        let http = reqwest::Client::new();

        println!("Starting MCP server (this might take a few seconds)...");

        // Connect to an MCP server via stdio.
        // Here we use the official Memory MCP server via NPX as an example.
        // (Requires Node.js/npx to be installed on your system).
        let mcp_tool = McpTool::stdio("npx", &["-y", "@modelcontextprotocol/server-memory"])
            .await?
            .with_timeout(Duration::from_secs(30));

        // Group the MCP tools into a ToolBundle
        // (A single MCP server can expose multiple tools)
        let mut bundle = ToolBundle::new();
        bundle.push(mcp_tool);

        println!("MCP server connected. Exposed tools:");
        for tool in bundle.raw_tools() {
            println!(" - {}", tool.function.name);
        }

        println!("\nSending request to Claude...");

        // Create the request and attach the MCP tool definitions
        let mut stream = Request::new(Provider::Anthropic, api_key)
            .system_prompt("You are a helpful AI assistant. You have access to a memory tool.")
            .user("Save the secret phrase 'RUST-IS-AWESOME' to memory, and then verify it by reading it back.")
            .tools(bundle.raw_tools())
            .stream(&http)
            .await?;

        println!("\nResponse stream:");

        while let Some(event) = stream.next().await {
            match event {
                LlmEvent::Token(t) => {
                    print!("{t}");
                }
                LlmEvent::ToolCall(tc) => {
                    println!("\n\n[Model is calling MCP tool: {}]", tc.name);
                    println!("Arguments: {}", tc.arguments);

                    // In a fully autonomous loop, you would execute the tool call
                    // and return the result back to the model as a new message.
                }
                LlmEvent::Done => break,
                LlmEvent::Error(e) => eprintln!("\nError: {e}"),
                _ => {}
            }
        }

        println!("\n");

        Ok(())
    }
}

#[cfg(feature = "mcp")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    mcp_example::run().await
}

// Dummy main for when the feature is disabled
#[cfg(not(feature = "mcp"))]
fn main() {
    println!("Please run this example with the `mcp` feature enabled:");
    println!("cargo run --example 05_mcp_client --features mcp");
}
