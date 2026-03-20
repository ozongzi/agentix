use std::marker::PhantomData;
use std::time::Duration;

use eventsource_stream::Eventsource;
use futures::{StreamExt, stream::BoxStream};
use reqwest::{Client, Response};

use tracing::{debug, instrument, warn};

use crate::error::{ApiError, Result};
use crate::request::Request;
use crate::types::ProviderProtocol;

/// Typed HTTP client bound to a specific provider `P`.
///
/// Use the type aliases [`DeepSeekAgent`][crate::DeepSeekAgent] etc. — they
/// each carry their own `ApiClient<P>` internally.  You only need to construct
/// this directly when building a custom provider or a low-level integration.
#[derive(Debug)]
pub struct ApiClient<P> {
    token: String,
    base_url: String,
    client: Client,
    timeout: Option<Duration>,
    _provider: PhantomData<P>,
}

impl<P> Clone for ApiClient<P> {
    fn clone(&self) -> Self {
        Self {
            token: self.token.clone(),
            base_url: self.base_url.clone(),
            client: self.client.clone(),
            timeout: self.timeout,
            _provider: PhantomData,
        }
    }
}

impl<P: ProviderProtocol> ApiClient<P> {
    /// Create a new client for provider `P` with the given token.
    /// The default base URL is taken from `P::default_base_url()` — each provider
    /// declares its own. Override it with [`with_base_url`][Self::with_base_url].
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            base_url: P::default_base_url().to_string(),
            client: Client::new(),
            timeout: None,
            _provider: PhantomData,
        }
    }

    /// Replace the base URL (builder style).
    pub fn with_base_url(mut self, base: impl Into<String>) -> Self {
        self.base_url = base.into();
        self
    }

    /// Replace the token (builder style).
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = token.into();
        self
    }

    /// Set an optional timeout applied to every request.
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = Some(t);
        self
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn provider_url(&self, model: &str, streaming: bool) -> String {
        format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            P::url_suffix(model, streaming)
        )
    }

    #[instrument(level = "debug", skip(self, raw), fields(url))]
    async fn post_json<T: serde::Serialize>(&self, url: &str, raw: &T) -> Result<Response> {
        debug!(method = "POST", %url, "sending request");

        let effective_url: String;
        let builder_url: &str;

        if P::uses_query_key_auth() {
            effective_url = if url.contains('?') {
                format!("{}&key={}", url, self.token)
            } else {
                format!("{}?key={}", url, self.token)
            };
            builder_url = &effective_url;
        } else {
            effective_url = url.to_string();
            builder_url = &effective_url;
        }

        let mut builder = self.client.post(builder_url);

        if !P::uses_query_key_auth() {
            match P::auth_header_name() {
                Some(header) => {
                    builder = builder.header(header, &self.token);
                }
                None => {
                    builder = builder.bearer_auth(&self.token);
                }
            }
        }

        for &(name, value) in P::extra_headers() {
            builder = builder.header(name, value);
        }

        builder = builder.json(raw);

        if let Some(t) = self.timeout {
            builder = builder.timeout(t);
            debug!(timeout_ms = t.as_millis(), "request timeout set");
        }

        let resp = builder.send().await.map_err(|e| {
            warn!(error = %e, "http send failed");
            ApiError::Reqwest(e)
        })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_else(|e| e.to_string());
            warn!(%status, body = %body, "non-success response");
            return Err(ApiError::http_error(status, body));
        }

        Ok(resp)
    }

    fn into_chunk_stream(resp: Response) -> BoxStream<'static, Result<P::RawChunk>> {
        resp.bytes_stream()
            .eventsource()
            .filter_map(|ev_res| async move {
                match ev_res {
                    Ok(ev) => {
                        if ev.data == "[DONE]" {
                            debug!("received [DONE] sentinel");
                            None
                        } else {
                            match serde_json::from_str::<P::RawChunk>(&ev.data) {
                                Ok(chunk) => Some(Ok(chunk)),
                                Err(e) => {
                                    warn!(data = %ev.data, "chunk parse failed");
                                    Some(Err(ApiError::Json(e)))
                                }
                            }
                        }
                    }
                    Err(e) => {
                        warn!(error = %e, "eventsource error");
                        Some(Err(ApiError::EventSource(e.to_string())))
                    }
                }
            })
            .boxed()
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Send a non-streaming request and return the parsed provider response.
    pub async fn send(&self, req: Request) -> Result<P::RawResponse> {
        let url = self.provider_url(&req.model, false);
        let raw = P::build_raw(req);
        let resp = self.post_json(&url, &raw).await?;
        resp.json::<P::RawResponse>()
            .await
            .map_err(ApiError::Reqwest)
    }

    /// Send a streaming request, consuming `self`, and return a `'static`
    /// `BoxStream` of parsed chunks.  Ownership is taken so the stream can be
    /// stored in a state machine without lifetime complications.
    pub async fn into_stream(
        self,
        req: Request,
    ) -> Result<BoxStream<'static, Result<P::RawChunk>>> {
        let url = self.provider_url(&req.model, true);
        let raw = P::build_raw(req);
        let resp = self.post_json(&url, &raw).await?;
        Ok(Self::into_chunk_stream(resp))
    }
}
