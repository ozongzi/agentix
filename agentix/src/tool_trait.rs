use crate::raw::shared::ToolDefinition as RawTool;
use async_trait::async_trait;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;
use futures::stream::BoxStream;

/// Output emitted by a tool during its execution.
pub enum ToolOutput {
    /// Intermediate progress (e.g. streaming stdout, reporting percentage).
    Progress(String),
    /// The final result of the tool execution.
    Result(Value),
}

/// The core trait that all agent tools must implement.
///
/// You should not implement this trait manually. Instead, annotate your `impl` block
/// with the [`#[tool]`][agentix_macros::tool] macro (for one-shot functions) or
/// [`#[streaming_tool]`][agentix_macros::streaming_tool] (for streaming output).
#[async_trait]
pub trait Tool: Send + Sync {
    /// Return the list of raw tool definitions to send to the API.
    fn raw_tools(&self) -> Vec<RawTool>;

    /// Invoke the named tool with the given arguments and return a stream of outputs.
    async fn call(&self, name: &str, args: Value) -> BoxStream<'static, ToolOutput>;
}

#[async_trait]
impl Tool for std::sync::Arc<dyn Tool> {
    fn raw_tools(&self) -> Vec<RawTool> {
        (**self).raw_tools()
    }
    async fn call(&self, name: &str, args: Value) -> BoxStream<'static, ToolOutput> {
        (**self).call(name, args).await
    }
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

    /// Remove the tool that provides the given function name.
    pub fn remove(&mut self, name: &str) {
        self.remove_by_names(&[name.to_string()]);
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

    /// Remove all tools from this bundle.
    pub fn clear(&mut self) {
        self.tools.clear();
        self.index.clear();
    }
}

#[async_trait]
impl Tool for ToolBundle {
    fn raw_tools(&self) -> Vec<RawTool> {
        let all: Vec<RawTool> = self.tools.iter().flat_map(|t| t.raw_tools()).collect();
        // Log duplicates before dedup so the root cause is visible in logs.
        {
            let mut counts = std::collections::HashMap::new();
            for r in &all {
                *counts.entry(r.function.name.as_str()).or_insert(0u32) += 1;
            }
            let dups: Vec<_> = counts.into_iter().filter(|(_, c)| *c > 1).collect();
            if !dups.is_empty() {
                eprintln!("[agentix] WARN duplicate tool names in ToolBundle: {:?}", dups);
            }
        }
        let mut seen = std::collections::HashSet::new();
        all.into_iter().filter(|r| seen.insert(r.function.name.clone())).collect()
    }

    async fn call(&self, name: &str, args: Value) -> BoxStream<'static, ToolOutput> {
        use futures::StreamExt;

        // Fast path: direct index lookup (covers tools registered at this level).
        if let Some(&idx) = self.index.get(name) {
            return self.tools[idx].call(name, args).await;
        }
        // Slow path: nested ToolBundle whose children aren't indexed at this level.
        for tool in &self.tools {
            if tool.raw_tools().iter().any(|r| r.function.name == name) {
                return tool.call(name, args).await;
            }
        }
        let err = json!({ "error": format!("unknown tool: {name}") });
        futures::stream::iter(vec![ToolOutput::Result(err)]).boxed()
    }
}

impl<T: Tool + 'static> std::ops::Add<T> for ToolBundle {
    type Output = ToolBundle;

    fn add(self, rhs: T) -> Self::Output {
        ToolBundle::with(self, rhs)
    }
}

impl<T: Tool + 'static> std::iter::Sum<T> for ToolBundle {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(ToolBundle::new(), |b, t| b + t)
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
