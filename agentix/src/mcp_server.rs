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
//! Expose tools over stdin/stdout. This is the standard way to integrate
//! with Claude Desktop:
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
//! ## HTTP server (Streamable HTTP)
//!
//! Expose tools over a real HTTP server (using SSE for async events):
//!
//! ```no_run
//! # use agentix::{McpServer, ToolBundle};
//! # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! McpServer::new(ToolBundle::new())
//!     .serve_http("0.0.0.0:3000")
//!     .await?;
//! # Ok(()) }
//! ```

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use rmcp::{
    model::{
        CallToolRequestParams, CallToolResult, ListToolsRequestParams, ListToolsResult,
        Tool as McpToolDef,
    },
    service::{Peer, RoleServer, Service},
    transport::{AxumHttpServerTransport, TokioChildProcess},
};
use serde_json::Value;
use tracing::{info, instrument};

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
    /// Create a new MCP server wrapping the given tool bundle.
    pub fn new(tools: ToolBundle) -> Self {
        Self {
            tools,
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
    ///
    /// This is the standard transport for local tools (e.g. used by Claude Desktop).
    pub async fn serve_stdio(self) -> Result<(), McpServerError> {
        info!(name = %self.name, version = %self.version, "starting MCP stdio server");
        let transport = TokioChildProcess::new_current_process()?;
        let service = McpService::new(self.tools, self.name, self.version);
        service
            .serve(transport)
            .await
            .map_err(|e| McpServerError::Service(e.to_string()))
    }

    /// Serve the tools over a Streamable HTTP server at the given address.
    ///
    /// Clients can connect via HTTP + SSE.
    pub async fn serve_http(self, addr: impl Into<SocketAddr>) -> Result<(), McpServerError> {
        let addr = addr.into();
        info!(name = %self.name, version = %self.version, %addr, "starting MCP HTTP server");

        let service = Arc::new(McpService::new(self.tools, self.name, self.version));
        let app = axum::Router::new().nest_service("/", self.into_http_service_with_arc(service));

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app)
            .await
            .map_err(|e| McpServerError::Bind(e))
    }

    /// Convert the server into an Axum-compatible `tower::Service`.
    ///
    /// Use this to nest the MCP server inside an existing Axum router.
    pub fn into_http_service(
        self,
        config: AxumHttpServerTransport<McpService>,
    ) -> AxumHttpServerTransport<McpService> {
        let service = McpService::new(self.tools, self.name, self.version);
        AxumHttpServerTransport::with_config(service, config)
    }

    /// Convert an Arc-wrapped service into an Axum-compatible `tower::Service`.
    fn into_http_service_with_arc(
        &self,
        service: Arc<McpService>,
    ) -> AxumHttpServerTransport<Arc<McpService>> {
        AxumHttpServerTransport::new(service)
    }
}

// ── McpService ────────────────────────────────────────────────────────────────

/// Internal implementation of the MCP `Service` trait.
struct McpService {
    tools: ToolBundle,
    name: String,
    version: String,
}

impl McpService {
    fn new(tools: ToolBundle, name: String, version: String) -> Self {
        Self {
            tools,
            name,
            version,
        }
    }
}

#[async_trait]
impl Service<RoleServer> for McpService {
    #[instrument(skip_all)]
    async fn initialize(
        &self,
        _peer: &Peer<RoleServer>,
        params: rmcp::model::InitializeRequestParams,
    ) -> Result<rmcp::model::InitializeResult, rmcp::Error> {
        info!(
            client_name = %params.client_info.name,
            client_version = %params.client_info.version,
            "MCP client connected"
        );

        Ok(rmcp::model::InitializeResult {
            protocol_version: "2024-11-05".into(),
            capabilities: rmcp::model::ServerCapabilities {
                tools: Some(rmcp::model::ToolCapabilities {
                    list_changed: Some(false),
                }),
                ..Default::default()
            },
            server_info: rmcp::model::Implementation {
                name: self.name.clone(),
                version: self.version.clone(),
            },
            instructions: None,
        })
    }

    #[instrument(skip_all, fields(name = %params.name))]
    async fn call_tool(
        &self,
        _peer: &Peer<RoleServer>,
        params: CallToolRequestParams,
    ) -> Result<CallToolResult, rmcp::Error> {
        let name = params.name.to_string();
        let arguments = Value::Object(params.arguments.unwrap_or_default());

        let result = self.tools.call(&name, arguments).await;

        Ok(CallToolResult {
            content: vec![rmcp::model::Content::Text {
                text: serde_json::to_string(&result).unwrap_or_default(),
            }],
            is_error: Some(result.is_object() && result.get("error").is_some()),
            ..Default::default()
        })
    }

    #[instrument(skip_all)]
    async fn list_tools(
        &self,
        _peer: &Peer<RoleServer>,
        _params: ListToolsRequestParams,
    ) -> Result<ListToolsResult, rmcp::Error> {
        let tools: Vec<McpToolDef> = self
            .tools
            .raw_tools()
            .into_iter()
            .map(|raw| McpToolDef {
                name: raw.function.name.into(),
                description: raw.function.description.map(Into::into),
                input_schema: Arc::new(raw.function.parameters.as_object().cloned().unwrap_or_default()),
            })
            .collect();

        Ok(ListToolsResult {
            tools,
            next_cursor: None,
            ..Default::default()
        })
    }
}

// Also implement for Arc<McpService> to support Axum transport
#[async_trait]
impl Service<RoleServer> for Arc<McpService> {
    async fn initialize(
        &self,
        peer: &Peer<RoleServer>,
        params: rmcp::model::InitializeRequestParams,
    ) -> Result<rmcp::model::InitializeResult, rmcp::Error> {
        self.as_ref().initialize(peer, params).await
    }
    async fn call_tool(
        &self,
        peer: &Peer<RoleServer>,
        params: CallToolRequestParams,
    ) -> Result<CallToolResult, rmcp::Error> {
        self.as_ref().call_tool(peer, params).await
    }
    async fn list_tools(
        &self,
        peer: &Peer<RoleServer>,
        params: ListToolsRequestParams,
    ) -> Result<ListToolsResult, rmcp::Error> {
        self.as_ref().list_tools(peer, params).await
    }
}
