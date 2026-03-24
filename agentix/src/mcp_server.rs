//! MCP (Model Context Protocol) **server** support for `agentix`.
//!
//! Enable with the `mcp-server` feature flag:
//!
//! ```toml
//! [dependencies]
//! agentix = { version = "0.4", features = ["mcp-server"] }
//! ```
//!
//! # Usage
//!
//! [`McpServer`] allows you to expose any [`ToolBundle`] (or individual [`Tool`]s)
//! as an MCP server. This is useful for building custom tool sets that can
//! be consumed by Claude Desktop, MCP Studio, or other `agentix` agents.
//!
//! ## Stdio server
//!
//! Expose tools over stdin/stdout:
//!
//! ```no_run
//! # use agentix::{McpServer, ToolBundle, tool};
//! # struct Calc;
//! # #[tool] impl agentix::Tool for Calc { async fn add(&self, a: f64, b: f64) -> f64 { a + b } }
//! # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! McpServer::new(ToolBundle::new().with(Calc))
//!     .serve_stdio()
//!     .await?;
//! # Ok(()) }
//! ```

use std::sync::Arc;

use futures::StreamExt;
use rmcp::{
    ErrorData as McpError,
    handler::server::ServerHandler,
    model::{
        CallToolRequestParams, CallToolResult, Content, Implementation,
        ListToolsResult, PaginatedRequestParams, ProgressNotificationParam,
        ServerCapabilities, ServerInfo, Tool as McpToolDef,
    },
    service::{RequestContext, RoleServer, serve_server},
};
use serde_json::{Value, json};
use tracing::info;

use crate::tool_trait::{Tool, ToolBundle};

// ── McpServerError ────────────────────────────────────────────────────────────

/// Errors that can occur while running an MCP server.
#[derive(Debug, thiserror::Error)]
pub enum McpServerError {
    #[error("failed to bind HTTP server: {0}")]
    Bind(#[from] std::io::Error),

    #[error("MCP service error: {0}")]
    Service(String),
}

// ── McpServer ─────────────────────────────────────────────────────────────────

/// An MCP server that exposes a collection of [`Tool`]s.
pub struct McpServer {
    tools: ToolBundle,
    name: String,
    version: String,
}

impl McpServer {
    /// Create a new MCP server wrapping the given tools.
    pub fn new(tools: impl Tool + 'static) -> Self {
        Self {
            tools: ToolBundle::new() + tools,
            name: "agentix-mcp-server".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        }
    }

    /// Set the server's name (sent to clients during handshake).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the server's version string.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Serve the tools over stdio (stdin/stdout).
    pub async fn serve_stdio(self) -> Result<(), McpServerError> {
        info!(name = %self.name, version = %self.version, "starting MCP stdio server");
        let handler = McpService::new(self.tools, self.name, self.version);
        let (rx, tx) = rmcp::transport::io::stdio();
        serve_server(handler, (rx, tx))
            .await
            .map_err(|e| McpServerError::Service(e.to_string()))?;
        Ok(())
    }

    /// Return an Axum [`Router`] that exposes the tools over HTTP using the
    /// MCP streamable-HTTP transport (POST + SSE).
    ///
    /// Mount this router at any path in your Axum application, for example:
    ///
    /// ```no_run
    /// # use agentix::{McpServer, ToolBundle};
    /// # async fn example() {
    /// let router = axum::Router::new()
    ///     .nest("/mcp", McpServer::new(ToolBundle::new()).into_axum_router());
    /// # }
    /// ```
    #[cfg(feature = "mcp-server")]
    pub fn into_axum_router(self) -> axum::Router {
        use rmcp::transport::{StreamableHttpService, StreamableHttpServerConfig};
        use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;

        let name = Arc::new(self.name);
        let version = Arc::new(self.version);
        let tools = Arc::new(self.tools);

        let service = StreamableHttpService::new(
            move || {
                let handler = McpService::new_shared(
                    Arc::clone(&tools),
                    (*name).clone(),
                    (*version).clone(),
                );
                Ok(handler)
            },
            Arc::new(LocalSessionManager::default()),
            StreamableHttpServerConfig::default(),
        );

        axum::Router::new().fallback_service(service)
    }
}

// ── McpService ────────────────────────────────────────────────────────────────

/// Internal implementation of the MCP `ServerHandler` trait.
struct McpService {
    tools: Arc<ToolBundle>,
    name: String,
    version: String,
}

impl McpService {
    fn new(tools: ToolBundle, name: String, version: String) -> Self {
        Self {
            tools: Arc::new(tools),
            name,
            version,
        }
    }

    fn new_shared(tools: Arc<ToolBundle>, name: String, version: String) -> Self {
        Self { tools, name, version }
    }
}

impl ServerHandler for McpService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(self.name.clone(), self.version.clone()))
    }

    fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CallToolResult, McpError>> + Send + '_ {
        let tools = Arc::clone(&self.tools);
        async move {
            use crate::tool_trait::ToolOutput;
            let name = request.name.to_string();
            let arguments = Value::Object(request.arguments.unwrap_or_default());

            // Extract progress token from request _meta if the client provided one.
            let progress_token = context.meta.get_progress_token();

            let mut stream = tools.call(&name, arguments).await;
            let mut final_result: Value = json!(null);
            let mut step: f64 = 0.0;

            while let Some(output) = stream.next().await {
                match output {
                    ToolOutput::Progress(msg) => {
                        if let Some(ref token) = progress_token {
                            step += 1.0;
                            let _ = context.peer.notify_progress(ProgressNotificationParam {
                                progress_token: token.clone(),
                                progress: step,
                                total: None,
                                message: Some(msg),
                            }).await;
                        }
                    }
                    ToolOutput::Result(v) => {
                        final_result = v;
                    }
                }
            }

            let is_error = final_result.is_object() && final_result.get("error").is_some();
            let text = serde_json::to_string(&final_result).unwrap_or_default();
            if is_error {
                Ok(CallToolResult::error(vec![Content::text(text)]))
            } else {
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
        }
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListToolsResult, McpError>> + Send + '_ {
        let tools: Vec<McpToolDef> = self
            .tools
            .raw_tools()
            .into_iter()
            .map(|raw| McpToolDef::new(
                raw.function.name,
                raw.function.description.unwrap_or_default(),
                raw.function
                    .parameters
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
            ))
            .collect();

        std::future::ready(Ok(ListToolsResult {
            tools,
            next_cursor: None,
            meta: None,
        }))
    }
}
