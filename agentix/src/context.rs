use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::Serialize;
use serde_json::Value;

// ── SharedContext ─────────────────────────────────────────────────────────────

/// A thread-safe, clone-cheap key/value store shared across graph nodes.
///
/// Nodes can read and write typed values by key.  [`PromptTemplate`] reads
/// context entries as template variables when a key is not found in its own
/// `.var()` map.
///
/// # Example
/// ```no_run
/// use agentix::{Graph, Node, PromptTemplate, SharedContext};
///
/// # #[tokio::main] async fn main() {
/// let ctx = SharedContext::new();
/// ctx.set("lang", "Japanese");
/// ctx.set("style", "formal");
///
/// let prompt = PromptTemplate::new("Translate {input} to {lang} in a {style} tone.")
///     .context(ctx.clone());
///
/// let agent = agentix::deepseek(std::env::var("KEY").unwrap());
/// Graph::new().edge(&prompt, &agent);
///
/// // Later, hot-swap the target language without rebuilding the graph:
/// ctx.set("lang", "Spanish");
/// # }
/// ```
///
/// [`PromptTemplate`]: crate::PromptTemplate
#[derive(Clone, Default)]
pub struct SharedContext(Arc<RwLock<HashMap<String, Value>>>);

impl SharedContext {
    pub fn new() -> Self { Self::default() }

    /// Insert or replace a value.  The value must be `serde::Serialize`.
    pub fn set(&self, key: impl Into<String>, value: impl Serialize) {
        if let Ok(v) = serde_json::to_value(value) {
            self.0.write().unwrap().insert(key.into(), v);
        }
    }

    /// Retrieve a value and deserialize it.  Returns `None` if the key is
    /// absent or deserialization fails.
    pub fn get<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        let map = self.0.read().unwrap();
        serde_json::from_value(map.get(key)?.clone()).ok()
    }

    /// Retrieve a value as a `String`.  JSON scalars are rendered as strings;
    /// objects/arrays are returned as compact JSON.
    pub fn get_str(&self, key: &str) -> Option<String> {
        let map = self.0.read().unwrap();
        Some(match map.get(key)? {
            Value::String(s) => s.clone(),
            other            => other.to_string(),
        })
    }

    /// Remove and return a value.
    pub fn remove(&self, key: &str) -> Option<Value> {
        self.0.write().unwrap().remove(key)
    }

    /// Iterate over all key/value pairs (snapshot).
    pub fn snapshot(&self) -> HashMap<String, Value> {
        self.0.read().unwrap().clone()
    }
}
