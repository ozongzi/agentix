//! Integration tests for the `mcp-server` feature.

#![cfg(feature = "mcp-server")]

use agentix::{McpServer, ToolBundle, tool, tool_trait::Tool};

// ── Test Tools ────────────────────────────────────────────────────────────────

struct CalcTool;

#[tool]
impl Tool for CalcTool {
    /// Add two numbers.
    /// a: first number
    /// b: second number
    async fn add(&self, a: f64, b: f64) -> f64 {
        a + b
    }

    /// Multiply two numbers.
    /// x: first number
    /// y: second number
    async fn multiply(&self, x: f64, y: f64) -> f64 {
        x * y
    }
}

struct EchoTool;

#[tool]
impl Tool for EchoTool {
    /// Echo a message.
    /// message: message to echo
    async fn echo(&self, message: String) -> String {
        format!("Echo: {message}")
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn mcp_server_creation_with_name_and_version() {
    let server = McpServer::new(CalcTool)
        .with_name("test-server")
        .with_version("1.0.0");

    let _ = server;
}

#[test]
fn mcp_server_with_tool_bundle() {
    let bundle = ToolBundle::new().with(CalcTool).with(EchoTool);

    let server = McpServer::new(bundle);
    let _ = server;
}

#[tokio::test]
async fn mcp_server_into_axum_router_creates_router() {
    let bundle = ToolBundle::new().with(CalcTool).with(EchoTool);

    let server = McpServer::new(bundle);
    let _router = server.into_axum_router();
}

// Test that multiple tools are correctly registered
#[test]
fn mcp_server_with_multiple_tools() {
    let bundle = ToolBundle::new().with(CalcTool).with(EchoTool);

    let raw_tools = bundle.raw_tools();

    // Should have 3 tools total (add, multiply, echo)
    assert_eq!(raw_tools.len(), 3, "Expected 3 tools: add, multiply, echo");

    let tool_names: Vec<&str> = raw_tools.iter().map(|t| t.function.name.as_str()).collect();

    assert!(tool_names.contains(&"add"), "Should have 'add' tool");
    assert!(
        tool_names.contains(&"multiply"),
        "Should have 'multiply' tool"
    );
    assert!(tool_names.contains(&"echo"), "Should have 'echo' tool");
}

// Test chained with_* methods
#[test]
fn mcp_server_builder_chaining() {
    let server = McpServer::new(CalcTool)
        .with_name("my-mcp-server")
        .with_version("2.0.0-beta");

    let _ = server;
}
