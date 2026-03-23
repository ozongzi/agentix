use crate::raw::shared::ToolDefinition as RawTool;
use async_trait::async_trait;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;

/// The core trait that all agent tools must implement.
///
/// You should not implement this trait manually. Instead, annotate your `impl` block
/// with the [`#[tool]`][agentix_macros::tool] macro and write plain `async fn` methods —
/// the macro generates the `raw_tools` and `call` implementations for you.
///
/// # What the macro generates
///
/// For each `async fn` in the annotated `impl`:
/// - A [`RawTool`] entry (name, description from doc comment, JSON Schema from parameter types)
///   is added to the `raw_tools()` vec.
/// - A `match` arm in `call()` that deserialises each argument from the incoming `args` JSON,
///   invokes the method, and serialises the return value via `serde_json::to_value`.
///
/// Any return type that implements `serde::Serialize` is accepted — `serde_json::Value`,
/// plain structs with `#[derive(Serialize)]`, primitives, `Option<T>`, `Vec<T>`, etc.
///
/// # Example
///
/// ```no_run
/// use agentix::tool;
///
/// struct Calc;
///
/// #[tool]
/// impl agentix::Tool for Calc {
///     /// Add two integers together.
///     /// a: first operand
///     /// b: second operand
///     async fn add(&self, a: i64, b: i64) -> i64 {
///         a + b
///     }
/// }
///
/// # #[tokio::main] async fn main() {
/// let agent = agentix::deepseek(std::env::var("DEEPSEEK_API_KEY").unwrap())
///     .tool(Calc);
/// # }
/// ```
#[async_trait]
pub trait Tool: Send + Sync {
    /// Return the list of raw tool definitions to send to the API.
    fn raw_tools(&self) -> Vec<RawTool>;

    /// Invoke the named tool with the given arguments and return the result as a JSON value.
    ///
    /// When using the `#[tool]` macro you do not implement this method yourself —
    /// the macro generates it. The generated implementation accepts any return type
    /// that implements `serde::Serialize` (including `serde_json::Value`, plain
    /// structs with `#[derive(Serialize)]`, primitives, etc.) and converts the
    /// value to `serde_json::Value` automatically.
    async fn call(&self, name: &str, args: Value) -> Value;
}

/// A collection of [`Tool`] implementations dispatched by name.
pub struct ToolBundle {
    tools: Vec<Box<dyn Tool>>,
    index: std::collections::HashMap<String, usize>,
}

impl Default for ToolBundle {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolBundle {
    pub fn new() -> Self {
        Self {
            tools: vec![],
            index: HashMap::new(),
        }
    }

    /// Add a tool to this bundle, returning `self` for chaining.
    pub fn with<T: Tool + 'static>(mut self, tool: T) -> Self {
        let idx = self.tools.len();
        for raw in tool.raw_tools() {
            self.index.insert(raw.function.name.clone(), idx);
        }
        self.tools.push(Box::new(tool));
        self
    }

    /// Add a tool to this bundle in-place.
    pub fn push<T: Tool + 'static>(&mut self, tool: T) {
        self.push_boxed(Box::new(tool));
    }

    /// Add an already-boxed tool in-place.
    pub fn push_boxed(&mut self, tool: Box<dyn Tool>) {
        let idx = self.tools.len();
        for raw in tool.raw_tools() {
            self.index.insert(raw.function.name.clone(), idx);
        }
        self.tools.push(tool);
    }

    /// Remove all tools whose `raw_tools()` contains any of the given names.
    pub fn remove_by_names(&mut self, names: &[String]) {
        let names_set: std::collections::HashSet<&str> =
            names.iter().map(String::as_str).collect();
        let mut new_tools: Vec<Box<dyn Tool>> = Vec::new();
        let mut new_index: HashMap<String, usize> = HashMap::new();
        for tool in self.tools.drain(..) {
            let raws = tool.raw_tools();
            if raws.iter().any(|r| names_set.contains(r.function.name.as_str())) {
                continue;
            }
            let idx = new_tools.len();
            for raw in raws {
                new_index.insert(raw.function.name.clone(), idx);
            }
            new_tools.push(tool);
        }
        self.tools = new_tools;
        self.index = new_index;
    }

    /// Iterate over the registered tools (read-only).
    pub fn tools(&self) -> impl Iterator<Item = &dyn Tool> {
        self.tools.iter().map(|t| t.as_ref())
    }

    /// Return the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns `true` if no tools have been registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[async_trait]
impl Tool for ToolBundle {
    fn raw_tools(&self) -> Vec<RawTool> {
        self.tools.iter().flat_map(|t| t.raw_tools()).collect()
    }

    async fn call(&self, name: &str, args: Value) -> Value {
        // Fast path: direct index lookup (covers tools registered at this level).
        if let Some(&idx) = self.index.get(name) {
            return self.tools[idx].call(name, args).await;
        }
        // Slow path: the name wasn't in the index, which can happen when a
        // nested ToolBundle was pushed (its children aren't individually indexed
        // at this level).  Ask each child that declares the name.
        for tool in &self.tools {
            if tool.raw_tools().iter().any(|r| r.function.name == name) {
                return tool.call(name, args).await;
            }
        }
        json!({ "error": format!("unknown tool: {name}") })
    }
}

impl<T: Tool + 'static> std::ops::Add<T> for ToolBundle {
    type Output = ToolBundle;

    fn add(self, rhs: T) -> Self::Output {
        ToolBundle::with(self, rhs)
    }
}

impl<T: Tool + 'static> std::ops::AddAssign<T> for ToolBundle {
    fn add_assign(&mut self, rhs: T) {
        self.push(rhs);
    }
}

// ── dtolnay trick (autoref specialization) ──────────────────────────────────

#[doc(hidden)]
pub trait ToolResultResult {
    fn __agentix_wrap(self) -> Value;
}

impl<T: serde::Serialize, E: std::fmt::Display> ToolResultResult for Result<T, E> {
    fn __agentix_wrap(self) -> Value {
        match self {
            Ok(v) => serde_json::to_value(v).unwrap_or_else(|e| {
                json!({ "error": format!("serialization error: {e}") })
            }),
            Err(e) => json!({ "error": e.to_string() }),
        }
    }
}

#[doc(hidden)]
pub trait ToolResultValue {
    fn __agentix_wrap(self) -> Value;
}

impl<T: serde::Serialize> ToolResultValue for &T {
    fn __agentix_wrap(self) -> Value {
        serde_json::to_value(self).unwrap_or_else(|e| {
            json!({ "error": format!("serialization error: {e}") })
        })
    }
}
