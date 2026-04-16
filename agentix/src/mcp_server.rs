//! MCP (Model Context Protocol) **server** support for `agentix`.
//!
//! Enable with the `mcp-server` feature flag:
//!
//! ```toml
//! [dependencies]
//! agentix = { version = "0.7", features = ["mcp-server"] }
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
//!
//! ## HTTP server
//!
//! Expose tools over HTTP (Streamable HTTP transport):
//!
//! ```no_run
//! # use agentix::{McpServer, ToolBundle, tool};
//! # struct Calc;
//! # #[tool] impl agentix::Tool for Calc { async fn add(&self, a: f64, b: f64) -> f64 { a + b } }
//! # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! McpServer::new(ToolBundle::new().with(Calc))
//!     .serve_http(("0.0.0.0", 3001))
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
use serde_json::Value;
use tracing::info;

use crate::request::{Content as AgentixContent, ImageData};
use crate::tool_trait::{Tool, ToolBundle};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Resolve an image URL into base64-encoded bytes. Handles `data:` URLs
/// locally (no network) and otherwise performs an HTTP GET via reqwest.
async fn fetch_image_as_base64(url: &str) -> Result<String, String> {
    use base64::{Engine, engine::general_purpose::STANDARD};

    if let Some(rest) = url.strip_prefix("data:") {
        // data:<mime>[;base64],<payload>
        if let Some((meta, payload)) = rest.split_once(',') {
            if meta.contains(";base64") {
                return Ok(payload.to_string());
            }
            return Ok(STANDARD.encode(payload.as_bytes()));
        }
        return Err("malformed data URL".into());
    }

    let bytes = reqwest::get(url)
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .bytes()
        .await
        .map_err(|e| e.to_string())?;
    Ok(STANDARD.encode(&bytes))
}

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
        let running = serve_server(handler, (rx, tx))
            .await
            .map_err(|e| McpServerError::Service(e.to_string()))?;
        running.waiting().await.ok();
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

    /// Bind a TCP listener and serve the tools over HTTP using the MCP
    /// streamable-HTTP transport (POST + SSE).
    ///
    /// `addr` accepts anything that [`tokio::net::TcpListener::bind`] accepts
    /// (e.g. `([0, 0, 0, 0], 3001)` or `"127.0.0.1:3001"`).
    ///
    /// This method blocks until the server is shut down.
    #[cfg(feature = "mcp-server")]
    pub async fn serve_http(self, addr: impl tokio::net::ToSocketAddrs) -> Result<(), McpServerError> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        let local_addr = listener.local_addr()?;
        info!(name = %self.name, version = %self.version, %local_addr, "starting MCP HTTP server");
        let router = self.into_axum_router();
        axum::serve(listener, router)
            .await
            .map_err(McpServerError::Bind)
    }
}

// ── McpService ────────────────────────────────────────────────────────────────

/// Internal implementation of the MCP `ServerHandler` trait.
pub struct McpService {
    tools: Arc<ToolBundle>,
    name: String,
    version: String,
}

impl McpService {
    pub fn new(tools: ToolBundle, name: String, version: String) -> Self {
        Self {
            tools: Arc::new(tools),
            name,
            version,
        }
    }

    pub fn new_shared(tools: Arc<ToolBundle>, name: String, version: String) -> Self {
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
            let mut final_result: Vec<AgentixContent> = Vec::new();
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

            let mut contents: Vec<Content> = Vec::with_capacity(final_result.len());
            for c in final_result {
                contents.push(match c {
                    AgentixContent::Text { text } => Content::text(text),
                    AgentixContent::Image(img) => {
                        let mime = img.mime_type;
                        match img.data {
                            ImageData::Base64(d) => Content::image(d, mime),
                            ImageData::Url(url) => match fetch_image_as_base64(&url).await {
                                Ok(b64) => Content::image(b64, mime),
                                Err(e) => Content::text(format!(
                                    "[image fetch failed: {url} ({e})]"
                                )),
                            },
                        }
                    }
                });
            }
            Ok(CallToolResult::success(contents))
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
