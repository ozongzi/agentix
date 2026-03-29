#[cfg(feature = "mcp-server")]
mod mcp_server_example {
    use agentix::{McpServer, ToolBundle, tool};

    // 1. Define your tools
    struct MathTools;

    #[tool]
    impl agentix::Tool for MathTools {
        /// Add two numbers together.
        /// a: first number
        /// b: second number
        async fn add(&self, a: f64, b: f64) -> f64 {
            a + b
        }

        /// Multiply two numbers.
        /// a: first number
        /// b: second number
        async fn multiply(&self, a: f64, b: f64) -> f64 {
            a * b
        }
    }

    pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
        // 2. Group the tools into a ToolBundle
        let bundle = ToolBundle::new().with(MathTools);

        // 3. Create the MCP Server instance
        let server = McpServer::new(bundle);

        // 4. Serve the tools!
        // We'll use HTTP (Streamable HTTP Transport) for this example so it doesn't
        // hijack your terminal's stdio, making it easier to see the server running.
        //
        // Note: For typical Claude Desktop or MCP Studio integration, you would
        // normally use `server.serve_stdio().await?` instead.
        let addr = ("127.0.0.1", 3001);
        println!("Starting MCP HTTP Server on http://{}:{}", addr.0, addr.1);
        println!("You can now connect an MCP client (using HTTP transport) to this endpoint.");
        println!("Press Ctrl+C to stop the server.");

        server.serve_http(addr).await?;

        Ok(())
    }
}

#[cfg(feature = "mcp-server")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    mcp_server_example::run().await
}

#[cfg(not(feature = "mcp-server"))]
fn main() {
    println!("Please run this example with the `mcp-server` feature enabled:");
    println!("cargo run --example 06_mcp_server --features mcp-server");
}
