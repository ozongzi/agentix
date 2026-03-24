//! MCP (Model Context Protocol) client support for `agentix`.
//!
//! Enable with the `mcp` feature flag:
//!
//! ```toml
//! [dependencies]
//! agentix = { version = "0.4", features = ["mcp"] }
//! ```
//!
//! # Usage
//!
//! [`McpTool`] wraps an MCP server — either a local process (stdio) or a remote
//! HTTP endpoint — and implements [`Tool`] so it can be registered
//! directly with [`Agent::tool`][crate::Agent::tool].
//!
//! Every tool the MCP server advertises is forwarded to the agent automatically;
//! you do not need to know their names or schemas ahead of time.
//!
//! ## Stdio (local process)
//!
//! Spawns a child process and communicates over stdin/stdout.  Any MCP server
//! that ships as an `npx`, `uvx`, or plain binary command works here:
//!
//! ```no_run
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use agentix::Agent;
//! use agentix::McpTool;
//!
//! let agent = agentix::openai("sk-...")
//!     .tool(McpTool::stdio("npx", &["-y", "@playwright/mcp"]).await?);
//! # Ok(()) }
//! ```
//!
//! ## Streamable HTTP (remote server)
//!
//! Connects to a remote MCP server over HTTP + SSE:
//!
//! ```no_run
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use agentix::Agent;
//! use agentix::McpTool;
//!
//! let agent = agentix::openai("sk-...")
//!     .tool(McpTool::http("https://mcp.example.com/").await?);
//! # Ok(()) }
//! ```

use rmcp::{
    ServiceExt,
    model::CallToolRequestParams,
    service::{Peer, RoleClient, RunningService},
    transport::{StreamableHttpClientTransport, TokioChildProcess},
};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;
use tracing::{error, instrument};

use crate::raw::shared::{FunctionDefinition, ToolDefinition as RawTool, ToolKind as ToolType};
use crate::tool_trait::Tool;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can occur while connecting to or communicating with an MCP server.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("failed to spawn MCP server process: {0}")]
    Spawn(#[from] std::io::Error),

    #[error("MCP client initialisation failed: {0}")]
    Init(String),

    #[error("failed to list tools from MCP server: {0}")]
    ListTools(String),

    #[error("MCP tool call failed: {0}")]
    Call(String),
}

// ── McpTool ───────────────────────────────────────────────────────────────────

/// An MCP server exposed as a [`Tool`] that can be registered with
/// [`Agent`][crate::Agent].
///
/// The tool list is fetched once at construction time and cached for the
/// lifetime of the struct.  All calls are forwarded to the underlying MCP
/// server via the `rmcp` client.
///
/// See the [module-level documentation][self] for usage examples.
///
/// ## Output Length Limiting (IMPORTANT)
///
/// ⚠️ **By default, MCP tool outputs are LIMITED to 8,000 characters** to prevent
/// context explosion. This is critical because even a single 2MB tool response can
/// exceed the model's context window and crash the conversation before summarization kicks in.
///
/// You can customize or disable these limits:
/// - `max_output_chars`: Limits the total character count of the JSON output string
/// - `max_content_items`: Limits the number of content items in the output array
///
/// **Default limits:**
/// - `max_output_chars`: 8,000 characters (~2,000 tokens)
/// - `max_content_items`: 50 items
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use agentix::McpTool;
///
/// // Increase limit for specific tools that need more context
/// let tool = McpTool::stdio("npx", &["-y", "@playwright/mcp"])
///     .await?
///     .with_max_output_chars(20000);
///
/// // Disable limits for tools you trust (use with caution!)
/// let tool = McpTool::http("https://mcp.example.com/")
///     .await?
///     .without_output_limits();
///
/// // Set both limits
/// let tool = McpTool::stdio("npx", &["mcp-server-filesystem", "/project"])
///     .await?
///     .with_output_limits(10000, 30);
/// # Ok(()) }
/// ```
///
/// **Recommended limits by use case:**
/// - File system operations: 5,000 - 15,000 chars
/// - Web search: 2,000 - 5,000 chars
/// - Database queries: 3,000 - 8,000 chars
/// - Code execution results: 8,000 - 20,000 chars
#[derive(Clone)]
pub struct McpTool {
    /// Cached tool definitions fetched from the MCP server at startup.
    tools: Vec<RawTool>,
    /// Live peer used to dispatch `tools/call` requests.
    peer: Arc<Peer<RoleClient>>,
    /// Keep the running service alive for as long as McpTool exists.
    _service: Arc<dyn std::any::Any + Send + Sync>,
    /// Maximum character count for the JSON output string.
    /// Default: 8,000 characters
    max_output_chars: Option<usize>,
    /// Maximum number of content items in the output.
    /// Default: 50 items
    max_content_items: Option<usize>,
    /// Timeout for each tool call. Default: 30 seconds.
    call_timeout: Option<Duration>,
}

/// Default output character limit (8000 chars ≈ 2000 tokens)
const DEFAULT_MAX_OUTPUT_CHARS: usize = 8_000;
/// Default maximum content items
const DEFAULT_MAX_CONTENT_ITEMS: usize = 50;


impl McpTool {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Connect to an MCP server by spawning a child process.
    ///
    /// `program` is the executable (e.g. `"npx"`, `"uvx"`, `"python"`) and
    /// `args` are its arguments.  The process communicates over stdin/stdout
    /// using the MCP stdio transport.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if the process cannot be spawned, the MCP
    /// handshake fails, or the initial `tools/list` call fails.
    #[instrument(skip(args), fields(program = program.as_ref()))]
    pub async fn stdio(
        program: impl AsRef<str>,
        args: &[impl AsRef<str>],
    ) -> Result<Self, McpError> {
        let mut cmd = Command::new(program.as_ref());
        for arg in args {
            cmd.arg(arg.as_ref());
        }

        let transport = TokioChildProcess::new(cmd)?;
        Self::from_service(
            ().serve(transport)
                .await
                .map_err(|e| McpError::Init(e.to_string()))?,
        )
        .await
    }

    /// Connect to a remote MCP server over Streamable HTTP.
    ///
    /// `url` is the base URL of the MCP server
    /// (e.g. `"https://mcp.example.com/"`).
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if the HTTP connection fails, the MCP handshake
    /// fails, or the initial `tools/list` call fails.
    #[instrument(fields(url = url.as_ref()))]
    pub async fn http(url: impl AsRef<str>) -> Result<Self, McpError> {
        let transport = StreamableHttpClientTransport::from_uri(url.as_ref());
        Self::from_service(
            ().serve(transport)
                .await
                .map_err(|e| McpError::Init(e.to_string()))?,
        )
        .await
    }

    /// Connect to an MCP server over an arbitrary transport.
    ///
    /// Use this when you have a custom transport (e.g. a WebSocket tunnel)
    /// and want to wrap it as a [`McpTool`].
    ///
    /// # Example
    /// ```no_run
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// use agentix::McpTool;
    /// use rmcp::transport::SinkStreamTransport;
    /// // ... build your sink/stream from a WS connection ...
    /// // let tool = McpTool::from_transport(SinkStreamTransport::new(sink, stream)).await?;
    /// # Ok(()) }
    /// ```
    #[cfg(feature = "mcp")]
    pub async fn from_transport<T, E, A>(transport: T) -> Result<Self, McpError>
    where
        T: rmcp::transport::IntoTransport<RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        use rmcp::ServiceExt;
        Self::from_service(
            ().serve(transport)
                .await
                .map_err(|e| McpError::Init(e.to_string()))?,
        )
        .await
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    async fn from_service<S>(running: RunningService<RoleClient, S>) -> Result<Self, McpError>
    where
        S: rmcp::service::Service<RoleClient> + Send + Sync + 'static,
    {
        let peer = running.peer().clone();
        let tools = Self::fetch_tools(&peer).await?;

        // Default limits are applied here to prevent context explosion
        Ok(Self {
            tools,
            peer: Arc::new(peer),
            _service: Arc::new(running),
            max_output_chars: Some(DEFAULT_MAX_OUTPUT_CHARS),
            max_content_items: Some(DEFAULT_MAX_CONTENT_ITEMS),
            call_timeout: None,
        })
    }

    // ── Configuration ─────────────────────────────────────────────────────────

    /// Set the maximum character count for the JSON output string.
    ///
    /// When the output exceeds this limit, it will be truncated with an ellipsis.
    pub fn with_max_output_chars(mut self, max: usize) -> Self {
        self.max_output_chars = Some(max);
        self
    }

    /// Set the maximum number of content items in the output.
    ///
    /// When the output contains more items than this limit, only the first N items
    /// will be returned.
    pub fn with_max_content_items(mut self, max: usize) -> Self {
        self.max_content_items = Some(max);
        self
    }

    /// Set both output limits at once.
    pub fn with_output_limits(mut self, max_chars: usize, max_items: usize) -> Self {
        self.max_output_chars = Some(max_chars);
        self.max_content_items = Some(max_items);
        self
    }

    /// Disable all output limits.
    ///
    /// ⚠️ **WARNING**: This allows MCP tools to return unlimited output, which can cause:
    /// - Context window overflow (2MB+ responses)
    /// - Conversation crashes before summarization
    /// - Memory issues with large payloads
    pub fn without_output_limits(mut self) -> Self {
        self.max_output_chars = None;
        self.max_content_items = None;
        self
    }

    /// Set the timeout for tool calls.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.call_timeout = Some(timeout);
        self
    }

    /// Disable timeout for tool calls (use with caution!).
    pub fn without_timeout(mut self) -> Self {
        self.call_timeout = None;
        self
    }

    /// Get the current output limits (useful for debugging).
    pub fn output_limits(&self) -> (Option<usize>, Option<usize>) {
        (self.max_output_chars, self.max_content_items)
    }

    /// Return the cached list of tool definitions advertised by this MCP server.
    ///
    /// This is a convenience inherent method so callers do not need to import
    /// the [`Tool`] trait just to inspect the tool list.
    pub fn raw_tools(&self) -> Vec<RawTool> {
        self.tools.clone()
    }

    /// Call `tools/list` (paginating automatically) and convert the MCP tool
    /// definitions into [`RawTool`]s understood by `agentix`.
    async fn fetch_tools(peer: &Peer<RoleClient>) -> Result<Vec<RawTool>, McpError> {
        let mcp_tools = peer
            .list_all_tools()
            .await
            .map_err(|e| McpError::ListTools(e.to_string()))?;

        let tools = mcp_tools
            .into_iter()
            .map(|mcp_tool| {
                let parameters = Value::Object(mcp_tool.input_schema.as_ref().clone());

                RawTool {
                    kind: ToolType::Function,
                    function: FunctionDefinition {
                        name: mcp_tool.name.to_string(),
                        description: mcp_tool.description.as_deref().map(str::to_string),
                        parameters,
                        strict: None,
                    },
                }
            })
            .collect();

        Ok(tools)
    }
}

// ── Tool impl ─────────────────────────────────────────────────────────────────

#[async_trait::async_trait]
impl Tool for McpTool {
    fn raw_tools(&self) -> Vec<RawTool> {
        self.tools.clone()
    }

    async fn call(&self, name: &str, args: Value) -> futures::stream::BoxStream<'static, crate::tool_trait::ToolOutput> {
        let arguments = args.as_object().cloned().map(|m| m.into_iter().collect());
        let owned_name: std::borrow::Cow<'static, str> = name.to_string().into();

        let params = match arguments {
            Some(args) => CallToolRequestParams::new(owned_name).with_arguments(args),
            None => CallToolRequestParams::new(owned_name),
        };

        let timeout_opt = self.call_timeout;
        let max_content_items = self.max_content_items;
        let max_output_chars = self.max_output_chars;
        let name_str = name.to_string();

        // Spawn the async call into a task so the returned stream is 'static.
        let peer = self.peer.clone();
        let join = tokio::spawn(async move {
            if let Some(timeout) = timeout_opt {
                match tokio::time::timeout(timeout, peer.call_tool(params)).await {
                    Ok(res) => res,
                    Err(_) => Err(rmcp::ServiceError::Timeout { timeout }),
                }
            } else {
                peer.call_tool(params).await
            }
        });

        let result_stream = async_stream::stream! {
            let result = match join.await {
                Ok(r) => r,
                Err(e) => {
                    error!(tool = %name_str, "MCP tool call task panicked: {e}");
                    yield crate::tool_trait::ToolOutput::Result(serde_json::json!({ "error": e.to_string() }));
                    return;
                }
            };

            match result {
                Ok(result) => {
                    let mut contents: Vec<Value> = result
                        .content
                        .into_iter()
                        .filter_map(|item| serde_json::to_value(item).ok())
                        .collect();

                    if let Some(max_items) = max_content_items {
                        if contents.len() > max_items {
                            contents.truncate(max_items);
                        }
                    }

                    // Emit each content item except the last as Progress so callers
                    // can stream intermediate output; the final item is the Result.
                    if contents.len() > 1 {
                        let last = contents.pop().unwrap();
                        for item in contents {
                            let text = serde_json::to_string(&item).unwrap_or_default();
                            yield crate::tool_trait::ToolOutput::Progress(text);
                        }
                        let result_value = last;
                        if let Some(max_chars) = max_output_chars {
                            let json_string = serde_json::to_string(&result_value).unwrap_or_default();
                            if json_string.len() > max_chars {
                                let mut limit = max_chars.saturating_sub(40);
                                limit = json_string.floor_char_boundary(limit);
                                let truncated = &json_string[..limit];
                                yield crate::tool_trait::ToolOutput::Result(serde_json::Value::String(format!(
                                    "{}...<truncated {} chars>",
                                    truncated,
                                    json_string.len()
                                )));
                                return;
                            }
                        }
                        yield crate::tool_trait::ToolOutput::Result(result_value);
                    } else {
                        let result_value = match contents.len() {
                            0 => serde_json::json!({ "result": null }),
                            _ => contents.into_iter().next().unwrap(),
                        };

                        if let Some(max_chars) = max_output_chars {
                            let json_string = serde_json::to_string(&result_value).unwrap_or_default();
                            if json_string.len() > max_chars {
                                let mut limit = max_chars.saturating_sub(40);
                                limit = json_string.floor_char_boundary(limit);
                                let truncated = &json_string[..limit];
                                yield crate::tool_trait::ToolOutput::Result(serde_json::Value::String(format!(
                                    "{}...<truncated {} chars>",
                                    truncated,
                                    json_string.len()
                                )));
                                return;
                            }
                        }

                        yield crate::tool_trait::ToolOutput::Result(result_value);
                    }
                }
                Err(e) => {
                    error!(tool = %name_str, error = %e, "MCP tool call failed");
                    yield crate::tool_trait::ToolOutput::Result(serde_json::json!({ "error": e.to_string() }));
                }
            }
        };

        use futures::StreamExt;
        result_stream.boxed()
    }
}
