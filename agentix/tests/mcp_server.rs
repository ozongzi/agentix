//! Integration tests for the `mcp-server` feature.

#![cfg(feature = "mcp-server")]

use agentix::{McpServer, ToolBundle, tool, tool_trait::Tool};
use futures::StreamExt as _;
use schemars::JsonSchema;
use serde::Serialize;

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

// ── isError / structuredContent / outputSchema ───────────────────────────────

#[derive(Serialize, JsonSchema)]
struct Stats {
    count: usize,
}

struct ResultTool;

#[tool]
impl Tool for ResultTool {
    /// Divide a by b.
    /// a: dividend
    /// b: divisor
    async fn divide(&self, a: f64, b: f64) -> Result<f64, String> {
        if b == 0.0 {
            Err("division by zero".into())
        } else {
            Ok(a / b)
        }
    }

    /// Return a struct.
    /// n: how many
    async fn stats(&self, n: usize) -> Stats {
        Stats { count: n }
    }

    /// Return opaque content blocks.
    async fn blocks(&self) -> Vec<agentix::Content> {
        vec![agentix::Content::text("hi")]
    }
}

async fn run(tool: &impl Tool, name: &str, args: serde_json::Value) -> agentix::ToolResult {
    let mut stream = tool.call(name, args).await;
    let mut last = agentix::ToolResult::ok(vec![]);
    while let Some(ev) = stream.next().await {
        if let Some(r) = ev.into_result() {
            last = r;
        }
    }
    last
}

#[tokio::test]
async fn err_variant_is_flagged_as_error() {
    let result = run(
        &ResultTool,
        "divide",
        serde_json::json!({"a": 1.0, "b": 0.0}),
    )
    .await;
    assert!(result.is_error, "Err(_) must surface as isError: true");
    assert!(result.structured.is_none());
}

#[tokio::test]
async fn bad_arguments_are_flagged_as_error() {
    let result = run(&ResultTool, "divide", serde_json::json!({"a": "nope"})).await;
    assert!(
        result.is_error,
        "argument deserialization failures must surface as isError: true"
    );
}

#[tokio::test]
async fn unknown_tool_is_flagged_as_error() {
    let bundle = ToolBundle::new().with(CalcTool);
    let result = run(&bundle, "nonexistent", serde_json::json!({})).await;
    assert!(result.is_error);
}

#[tokio::test]
async fn ok_variant_carries_structured_content() {
    let result = run(
        &ResultTool,
        "divide",
        serde_json::json!({"a": 6.0, "b": 2.0}),
    )
    .await;
    assert!(!result.is_error);
    // Scalars are wrapped so structuredContent is always an object.
    assert_eq!(result.structured, Some(serde_json::json!({"result": 3.0})));
}

#[tokio::test]
async fn struct_return_is_structured_verbatim() {
    let result = run(&ResultTool, "stats", serde_json::json!({"n": 7})).await;
    assert_eq!(result.structured, Some(serde_json::json!({"count": 7})));
}

#[tokio::test]
async fn content_returns_have_no_structured_payload() {
    let result = run(&ResultTool, "blocks", serde_json::json!({})).await;
    assert!(!result.is_error);
    assert!(
        result.structured.is_none(),
        "Vec<Content> is opaque and must not be structured"
    );
}

#[test]
fn output_schemas_wrap_scalars_and_pass_objects_through() {
    let schemas = ResultTool.output_schemas();

    // `Result<f64, _>` — schema of the Ok type, wrapped because it is a scalar.
    assert_eq!(
        schemas.get("divide"),
        Some(&serde_json::json!({
            "type": "object",
            "properties": { "result": { "type": "number", "format": "double" } },
            "required": ["result"],
        }))
    );

    // A struct is already an object, so it is used as-is.
    let stats = schemas.get("stats").expect("stats needs an output schema");
    assert_eq!(stats["type"], "object");
    assert_eq!(stats["properties"]["count"]["type"], "integer");

    // `Vec<Content>` has no JsonSchema and must not claim one.
    assert!(
        !schemas.contains_key("blocks"),
        "opaque content must not declare an outputSchema"
    );
}

#[test]
fn bundles_merge_output_schemas_from_children() {
    let bundle = ToolBundle::new().with(CalcTool).with(ResultTool);
    let schemas = bundle.output_schemas();
    assert!(schemas.contains_key("add"));
    assert!(schemas.contains_key("divide"));
    assert!(schemas.contains_key("stats"));
}
