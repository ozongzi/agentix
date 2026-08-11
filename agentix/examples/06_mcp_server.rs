//! Example 06: Expose your tools as an MCP server
//!
//! Anything that is a [`Tool`] can be served over the Model Context Protocol,
//! so Claude Desktop, MCP Studio, or another agentix agent can call it. This
//! example serves one stateless tool and one stateful tool bundle together.
//!
//! Run with:
//!   cargo run --example 06_mcp_server --features mcp-server
//!
//! Then point an MCP client at http://127.0.0.1:3001, or poke it by hand:
//!   curl -sN -X POST http://127.0.0.1:3001/ \
//!     -H 'Content-Type: application/json' \
//!     -H 'Accept: application/json, text/event-stream' \
//!     -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
//!          "protocolVersion":"2025-06-18","capabilities":{},
//!          "clientInfo":{"name":"curl","version":"1"}}}'

use agentix::{McpServer, McpServerError, tool};
use schemars::JsonSchema;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Mutex;

// ── Stateless: a free function is already a tool ──────────────────────────────
//
// The doc comment becomes the tool description, `a:`/`b:` lines become the
// parameter descriptions, and the argument types become the input schema.

/// Add two numbers together.
/// a: first number
/// b: second number
#[tool]
async fn add(a: f64, b: f64) -> f64 {
    a + b
}

// ── Stateful: several tools sharing one struct ────────────────────────────────
//
// `Tool::call` takes `&self`, so state you own and mutate needs interior
// mutability — a `Mutex`, `RwLock`, or `DashMap`. Dependencies that are already
// shareable (a `reqwest::Client`, a `sqlx::Pool`) go in as plain fields.
//
// No `Arc` is needed: `McpServer` holds a single instance and shares it across
// every client session, so all callers see the same notes.

#[derive(Default)]
struct Notes {
    entries: Mutex<HashMap<String, String>>,
}

/// Returned by `list_notes`. Deriving `JsonSchema` gives the tool an MCP
/// `outputSchema`, and its results arrive as `structuredContent` — so clients
/// get a typed payload instead of having to parse the text back out.
#[derive(Serialize, JsonSchema)]
struct NoteIndex {
    /// Keys of every stored note, sorted.
    keys: Vec<String>,
    /// How many notes are stored.
    count: usize,
}

#[tool]
impl agentix::Tool for Notes {
    /// Save a note, replacing any existing note under the same key.
    /// key: name to file the note under
    /// text: the note body
    async fn save_note(&self, key: String, text: String) -> String {
        self.entries.lock().unwrap().insert(key.clone(), text);
        format!("saved '{key}'")
    }

    /// Read a note back.
    /// key: name the note was filed under
    async fn read_note(&self, key: String) -> Result<String, String> {
        // `Err` reaches the client as `isError: true`, which is how the model
        // is told the call failed rather than returned the string "no note…".
        self.entries
            .lock()
            .unwrap()
            .get(&key)
            .cloned()
            .ok_or_else(|| format!("no note named '{key}'"))
    }

    /// List the keys of every stored note.
    async fn list_notes(&self) -> NoteIndex {
        let entries = self.entries.lock().unwrap();
        let mut keys: Vec<String> = entries.keys().cloned().collect();
        keys.sort();
        NoteIndex {
            count: keys.len(),
            keys,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), McpServerError> {
    // `+` combines tools. A single tool can also be passed on its own —
    // `McpServer::new(add)` is a complete server.
    let server = McpServer::new(add + Notes::default())
        .with_name("agentix-example")
        .with_version(env!("CARGO_PKG_VERSION"));

    // Claude Desktop and MCP Studio spawn servers over stdio — that's
    // `server.serve_stdio().await`. HTTP is used here so the example doesn't
    // take over your terminal and you can curl it.
    println!("MCP server listening on http://127.0.0.1:3001 — Ctrl+C to stop");
    server.serve_http(("127.0.0.1", 3001)).await
}
