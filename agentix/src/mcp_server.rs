//! MCP (Model Context Protocol) **server** support for `ds-api`.
//!
//! Enable with the `mcp-server` feature flag:
//!
//! ```toml
//! [dependencies]
//! ds-api = { version = "0.10", features = ["mcp-server"] }
//! ```
//!
//! # Overview
//!
//! [`McpServer`] wraps a [`ToolBundle`] and exposes it as a fully-compliant MCP
//! server.  Two transports are supported:
//!
//! ## Stdio (Claude Desktop / MCP Studio)
//!
//! Reads JSON-RPC messages from stdin and writes responses to stdout.
//! This is the transport used by Claude Desktop's MCP integration.
//!
//! ```no_run
//! use agentix::{ToolBundle, tool, McpServer};
//!
//! struct Greeter;
//!
//! #[tool]
//! impl agentix::Tool for Greeter {
//!     /// Say hello to someone.
//!     /// name: person's name
//!     async fn greet(&self, name: String) -> String {
//!         format!("Hello, {name}!")
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     McpServer::new(ToolBundle::new().add(Greeter))
//!         .serve_stdio()
//!         .await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Streamable HTTP
//!
//! Listens on a TCP address and serves the
//! [MCP Streamable HTTP transport](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports#streamable-http).
//!
//! ```no_run
//! use agentix::{ToolBundle, tool, McpServer};
//!
//! struct Greeter;
//!
//! #[tool]
//! impl agentix::Tool for Greeter {
//!     /// Say hello to someone.
//!     /// name: person's name
//!     async fn greet(&self, name: String) -> String {
//!         format!("Hello, {name}!")
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     McpServer::new(ToolBundle::new().add(Greeter))
//!         .serve_http("127.0.0.1:3000")
//!         .await?;
//!     Ok(())
//! }
//! ```
//!
//! The HTTP server mounts the MCP endpoint at `/mcp`.
//! For custom routing, use [`McpServer::into_http_service`] to obtain the
//! underlying [`StreamableHttpService`] and integrate it with your own Axum
//! (or any Tower-compatible) router.

use std::sync::Arc;

use rmcp::{
    RoleServer, ServerHandler, ServiceExt,
    model::{
        CallToolRequestParams, CallToolResult, Implementation, ListToolsResult,
        PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool as RmcpTool,
        ToolsCapability,
    },
    service::RequestContext,
    transport::streamable_http_server::{
        session::local::LocalSessionManager,
        tower::{StreamableHttpService, StreamableHttpServerConfig},
    },
};
use serde_json::Value;

use crate::tool_trait::Tool;
use crate::tool_trait::ToolBundle;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can occur while starting or running an MCP server.
#[derive(Debug, thiserror::Error)]
pub enum McpServerError {
    #[error("failed to bind address: {0}")]
    Bind(#[from] std::io::Error),

    #[error("server error: {0}")]
    Serve(String),
}

// ── Internal handler ──────────────────────────────────────────────────────────

/// rmcp `ServerHandler` that delegates all tool calls to a shared `ToolBundle`.
#[derive(Clone)]
pub struct BundleHandler {
    bundle: Arc<ToolBundle>,
    name: Arc<str>,
    version: Arc<str>,
}

impl ServerHandler for BundleHandler {
    fn get_info(&self) -> ServerInfo {
        let mut capabilities = ServerCapabilities::default();
        capabilities.tools = Some(ToolsCapability::default());
        ServerInfo::new(capabilities)
            .with_server_info(Implementation::new(&*self.name, &*self.version))
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, rmcp::ErrorData> {
        let tools = self
            .bundle
            .raw_tools()
            .into_iter()
            .map(|raw| {
                let input_schema: Arc<serde_json::Map<String, Value>> = match raw.function.parameters {
                    Value::Object(map) => Arc::new(map),
                    _ => Arc::new(serde_json::Map::new()),
                };
                RmcpTool::new_with_raw(
                    raw.function.name,
                    raw.function.description.map(Into::into),
                    input_schema,
                )
            })
            .collect();

        Ok(ListToolsResult::with_all_items(tools))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let args = request
            .arguments
            .map(Value::Object)
            .unwrap_or(Value::Object(serde_json::Map::new()));

        let result = self.bundle.call(&request.name, args).await;

        let is_error = result
            .as_object()
            .and_then(|o| o.get("error"))
            .is_some();

        if is_error {
            Ok(CallToolResult::structured_error(result))
        } else {
            Ok(CallToolResult::structured(result))
        }
    }
}

// ── McpServer ─────────────────────────────────────────────────────────────────

/// Wraps a [`ToolBundle`] and serves it as an MCP server.
///
/// See the [module-level documentation][self] for usage examples.
pub struct McpServer {
    bundle: Arc<ToolBundle>,
    name: Arc<str>,
    version: Arc<str>,
}

impl McpServer {
    /// Create a new `McpServer` backed by `bundle`.
    pub fn new(bundle: ToolBundle) -> Self {
        Self {
            bundle: Arc::new(bundle),
            name: "ds-api-mcp-server".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        }
    }

    /// Override the server name advertised during the MCP handshake.
    pub fn with_name(mut self, name: impl Into<Arc<str>>) -> Self {
        self.name = name.into();
        self
    }

    /// Override the server version advertised during the MCP handshake.
    pub fn with_version(mut self, version: impl Into<Arc<str>>) -> Self {
        self.version = version.into();
        self
    }

    fn make_handler(&self) -> BundleHandler {
        BundleHandler {
            bundle: self.bundle.clone(),
            name: self.name.clone(),
            version: self.version.clone(),
        }
    }

    // ── Stdio transport ───────────────────────────────────────────────────────

    /// Run the MCP server over **stdin / stdout**.
    ///
    /// This is the transport expected by Claude Desktop and MCP Studio.
    /// The call blocks until the client disconnects.
    ///
    /// # Errors
    ///
    /// Returns [`McpServerError`] if the server fails to start or encounters
    /// a fatal error while running.
    pub async fn serve_stdio(self) -> Result<(), McpServerError> {
        let handler = self.make_handler();
        let (stdin, stdout) = rmcp::transport::stdio();
        handler
            .serve((stdin, stdout))
            .await
            .map_err(|e| McpServerError::Serve(e.to_string()))?
            .waiting()
            .await
            .map_err(|e| McpServerError::Serve(e.to_string()))?;
        Ok(())
    }

    // ── HTTP transport ────────────────────────────────────────────────────────

    /// Run the MCP server over **Streamable HTTP**, listening on `addr`.
    ///
    /// The MCP endpoint is mounted at `/mcp`.  For custom routing, use
    /// [`McpServer::into_http_service`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`McpServerError`] if the address cannot be bound or if the
    /// HTTP server encounters a fatal error.
    pub async fn serve_http(self, addr: &str) -> Result<(), McpServerError> {
        let service = self.into_http_service(Default::default());
        let router = axum::Router::new().nest_service("/mcp", service);
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, router)
            .await
            .map_err(|e| McpServerError::Serve(e.to_string()))?;
        Ok(())
    }

    /// Convert into a Tower-compatible [`StreamableHttpService`].
    ///
    /// Use this when you need to integrate the MCP service into an existing
    /// Axum (or other Tower-based) router:
    ///
    /// ```no_run
    /// use agentix::{ToolBundle, McpServer};
    /// use rmcp::transport::streamable_http_server::tower::StreamableHttpServerConfig;
    ///
    /// # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let service = McpServer::new(ToolBundle::new())
    ///     .into_http_service(Default::default());
    ///
    /// let router = axum::Router::new()
    ///     .nest_service("/mcp", service)
    ///     .route("/health", axum::routing::get(|| async { "ok" }));
    ///
    /// let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
    /// axum::serve(listener, router).await?;
    /// # Ok(()) }
    /// ```
    pub fn into_http_service(
        self,
        config: StreamableHttpServerConfig,
    ) -> StreamableHttpService<BundleHandler, LocalSessionManager> {
        let handler = self.make_handler();
        StreamableHttpService::new(move || Ok(handler.clone()), Default::default(), config)
    }
}
