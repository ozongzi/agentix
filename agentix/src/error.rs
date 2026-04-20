use reqwest::StatusCode;
use thiserror::Error;

/// Common `Result` alias used throughout the crate.
pub type Result<T> = std::result::Result<T, ApiError>;

/// Unified error type covering LLM providers, network, and tool execution.
#[derive(Error, Debug)]
pub enum ApiError {
    /// HTTP-level failure from the provider.
    #[error("HTTP {status}: {text}")]
    Http { status: StatusCode, text: String },

    /// Business-level error returned by the LLM provider (e.g., policy violation, insufficient balance).
    #[error("LLM Provider error: {0}")]
    Llm(String),

    /// Network or request-level error from the underlying HTTP client.
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// Error during JSON serialization or deserialization.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Error during tool execution or parameter validation.
    #[error("Tool error: {0}")]
    Tool(String),

    /// Error in a Model Context Protocol (MCP) operation.
    #[error("MCP error: {0}")]
    Mcp(String),

    /// Failure in the EventSource / SSE stream.
    #[error("Stream error: {0}")]
    Stream(String),

    /// Invalid configuration or missing required parameters.
    #[error("Configuration error: {0}")]
    Config(String),

    /// General IO failure.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Catch-all for other error types.
    #[error("{0}")]
    Other(String),
}

impl ApiError {
    /// Returns true if the error is likely transient and should be retried.
    pub fn is_retriable(&self) -> bool {
        match self {
            ApiError::Http { status, .. } => {
                // Common retriable HTTP status codes
                matches!(
                    *status,
                    StatusCode::TOO_MANY_REQUESTS | // 429
                    StatusCode::INTERNAL_SERVER_ERROR | // 500
                    StatusCode::BAD_GATEWAY | // 502
                    StatusCode::SERVICE_UNAVAILABLE | // 503
                    StatusCode::GATEWAY_TIMEOUT // 504
                )
            }
            ApiError::Network(e) => {
                // Retry on connect/timeout errors
                e.is_connect() || e.is_timeout()
            }
            _ => false,
        }
    }

    /// Construct an [`ApiError::Http`] from a status code and body text.
    pub fn http(status: StatusCode, text: impl Into<String>) -> Self {
        Self::Http {
            status,
            text: text.into(),
        }
    }
}

impl From<&str> for ApiError {
    fn from(s: &str) -> Self {
        ApiError::Other(s.to_string())
    }
}

impl From<String> for ApiError {
    fn from(s: String) -> Self {
        ApiError::Other(s)
    }
}
