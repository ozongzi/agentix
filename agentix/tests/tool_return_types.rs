//! Integration tests verifying that `#[tool]` accepts any return type that
//! implements `serde::Serialize`.
//!
//! These live in `tests/` (integration test harness) so the generated code's
//! `agentix::` paths resolve correctly.

use agentix::{tool, ToolOutput};
use agentix::tool_trait::Tool;
use serde::Serialize;
use serde_json::{Value, json};
use futures::StreamExt;

// ── Tool definitions ──────────────────────────────────────────────────────────

/// Returns a raw `serde_json::Value` — the original behaviour.
struct ValueTool;

#[tool]
impl Tool for ValueTool {
    async fn run(&self, input: String) -> Value {
        json!({ "echo": input })
    }
}

/// Returns a plain `String`.
struct StringTool;

#[tool]
impl Tool for StringTool {
    async fn run(&self, input: String) -> String {
        format!("hello, {input}")
    }
}

/// Returns a primitive integer.
struct IntTool;

#[tool]
impl Tool for IntTool {
    async fn run(&self, x: i64, y: i64) -> i64 {
        x + y
    }
}

/// Returns a `bool`.
struct BoolTool;

#[tool]
impl Tool for BoolTool {
    async fn run(&self, value: bool) -> bool {
        !value
    }
}

/// A custom struct that derives `Serialize`.
#[derive(Serialize)]
struct WeatherReport {
    city: String,
    temp_c: f64,
    sunny: bool,
}

struct WeatherTool;

#[tool]
impl Tool for WeatherTool {
    /// Get a weather report for a city.
    /// city: name of the city
    async fn report(&self, city: String) -> WeatherReport {
        WeatherReport {
            city,
            temp_c: 22.5,
            sunny: true,
        }
    }
}

/// Returns `Option<String>` — `None` serializes to JSON `null`.
struct MaybeTool;

#[tool]
impl Tool for MaybeTool {
    async fn run(&self, present: bool) -> Option<String> {
        if present {
            Some("yes".to_string())
        } else {
            None
        }
    }
}

/// Returns a `Vec` of a serializable type.
struct ListTool;

#[tool]
impl Tool for ListTool {
    async fn run(&self, n: u32) -> Vec<u32> {
        (0..n).collect::<Vec<u32>>()
    }
}

/// Returns a `Result<T, E>` — `Ok` serializes to `T`, `Err` to `{ "error": E.to_string() }`.
struct ResultTool;

#[tool]
impl Tool for ResultTool {
    async fn run(&self, success: bool) -> Result<String, String> {
        if success {
            Ok("success".to_string())
        } else {
            Err("failed".to_string())
        }
    }
}

/// Returns a `Result<T, anyhow::Error>` (or similar) to test Display-based error formatting.
struct AnyhowResultTool;

#[tool]
impl Tool for AnyhowResultTool {
    async fn run(&self, success: bool) -> Result<i32, std::io::Error> {
        if success {
            Ok(42)
        } else {
            Err(std::io::Error::other("io error"))
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async fn call(tool: &impl Tool, name: &str, args: Value) -> Value {
    let mut stream = tool.call(name, args).await;
    let mut last_res = Value::Null;
    while let Some(ev) = stream.next().await {
        if let ToolOutput::Result(v) = ev {
            last_res = v;
        }
    }
    last_res
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn value_tool_returns_json_object() {
    let result = call(&ValueTool, "run", json!({ "input": "world" })).await;
    assert_eq!(result, json!({ "echo": "world" }));
}

#[tokio::test]
async fn string_tool_returns_json_string() {
    let result = call(&StringTool, "run", json!({ "input": "Alice" })).await;
    assert_eq!(result, Value::String("hello, Alice".to_string()));
}

#[tokio::test]
async fn int_tool_returns_json_number() {
    let result = call(&IntTool, "run", json!({ "x": 3, "y": 4 })).await;
    assert_eq!(result, json!(7_i64));
}

#[tokio::test]
async fn bool_tool_returns_json_bool() {
    let result = call(&BoolTool, "run", json!({ "value": true })).await;
    assert_eq!(result, json!(false));
}

#[tokio::test]
async fn struct_tool_serializes_fields() {
    let result = call(&WeatherTool, "report", json!({ "city": "Tokyo" })).await;
    assert_eq!(result["city"], json!("Tokyo"));
    assert_eq!(result["temp_c"], json!(22.5_f64));
    assert_eq!(result["sunny"], json!(true));
}

#[tokio::test]
async fn maybe_tool_some_returns_string() {
    let result = call(&MaybeTool, "run", json!({ "present": true })).await;
    assert_eq!(result, json!("yes"));
}

#[tokio::test]
async fn maybe_tool_none_returns_null() {
    let result = call(&MaybeTool, "run", json!({ "present": false })).await;
    assert_eq!(result, Value::Null);
}

#[tokio::test]
async fn list_tool_returns_json_array() {
    let result = call(&ListTool, "run", json!({ "n": 4 })).await;
    assert_eq!(result, json!([0, 1, 2, 3]));
}

#[tokio::test]
async fn unknown_tool_name_returns_error_json() {
    let result = call(&ValueTool, "nonexistent", json!({})).await;
    assert!(result["error"].as_str().unwrap().contains("unknown tool"));
}

#[tokio::test]
async fn bad_argument_type_returns_error_json() {
    // Pass a string where an integer is expected.
    let result = call(&IntTool, "run", json!({ "x": "not_a_number", "y": 1 })).await;
    assert!(
        result["error"]
            .as_str()
            .unwrap()
            .contains("invalid argument")
    );
}

#[tokio::test]
async fn result_tool_ok_returns_string() {
    let result = call(&ResultTool, "run", json!({ "success": true })).await;
    assert_eq!(result, json!("success"));
}

#[tokio::test]
async fn result_tool_err_returns_error_object() {
    let result = call(&ResultTool, "run", json!({ "success": false })).await;
    assert_eq!(result, json!({ "error": "failed" }));
}

#[tokio::test]
async fn result_tool_with_typed_error_returns_stringified_error() {
    let result = call(&AnyhowResultTool, "run", json!({ "success": false })).await;
    assert_eq!(result, json!({ "error": "io error" }));
}

#[tokio::test]
async fn result_tool_with_typed_error_returns_ok_value() {
    let result = call(&AnyhowResultTool, "run", json!({ "success": true })).await;
    assert_eq!(result, json!(42));
}
