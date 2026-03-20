//! Unified error types
//!
//! This module uses `thiserror` to provide a unified `ApiError` type for the crate and exports a
//! convenient `Result<T>` alias. The goal is to centralize disparate error sources (for example,
//! `Box<dyn Error>`, `reqwest::Error`, `serde_json::Error`) into a single `ApiError` so that `?`
//! conversions are simpler and error messages are more consistent.

use thiserror::Error;

/// Unified error type covering common error sources and including a generic string variant for easy conversions.
#[derive(Error, Debug)]
pub enum ApiError {
    /// HTTP-level failure (useful when preserving status code and response body text).
    #[error("HTTP error {status}: {text}")]
    Http {
        status: reqwest::StatusCode,
        text: String,
    },

    /// Network/request error from reqwest.
    #[error("Reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),

    /// JSON (serde_json) serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// EventSource / SSE handling error (originating from the `eventsource-stream` crate).
    /// Stored as a string to avoid depending on the concrete error type signature.
    #[error("EventSource error: {0}")]
    EventSource(String),

    /// IO error (fallback).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Generic string error (convenient for converting from `String` / `&str`).
    #[error("{0}")]
    Other(String),

    /// Unknown or placeholder error.
    #[error("Unknown error")]
    Unknown,
}

/// Common `Result` alias used throughout the crate.
pub type Result<T> = std::result::Result<T, ApiError>;

impl ApiError {
    /// Convenience constructor for the `Http` variant.
    pub fn http_error(status: reqwest::StatusCode, text: impl Into<String>) -> Self {
        ApiError::Http {
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
