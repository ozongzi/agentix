//! Raw `claude-code` provider — drives `claude -p` as a single-turn LLM and
//! emits `LlmEvent`s / returns a `CompleteResponse`, matching every other
//! raw provider in this crate.
//!
//! # How it works
//!
//! 1. Spawn an in-process MCP server whose tools are schema-only **stubs**
//!    (see [`StubTools`]). The caller's [`ToolDefinition`]s are surfaced so
//!    the model can emit `tool_use` blocks, but `call()` returns an empty
//!    result instantly — the caller dispatches tool calls externally.
//! 2. Spawn `claude -p --input-format stream-json --output-format stream-json`
//!    connected to that MCP server over loopback HTTP.
//! 3. Feed the last user message on stdin, parse stream-json lines on stdout.
//! 4. On the first `assistant` message, flush final `ToolCall`s + `Usage`,
//!    yield `Done`, and kill the subprocess. Further turns (tool dispatch,
//!    follow-up) are the caller's responsibility.
//!
//! Auth comes from the local `claude` CLI (Max OAuth / keychain); `api_key`
//! is ignored.

pub(crate) mod session;

use std::process::Stdio;

use async_stream::stream;
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::process::{Child, Command};
use tracing::{debug, info, warn};

use crate::config::AgentConfig;
use crate::error::ApiError;
use crate::mcp_server::McpServer;
use crate::msg::LlmEvent;
use crate::raw::shared::ToolDefinition;
use crate::request::{Message, ToolCall};
use crate::tool_trait::{Tool, ToolOutput};
use crate::types::{CompleteResponse, FinishReason, PartialToolCall, ToolCallChunk, UsageStats};

use self::session::{
    Cleanup, MCP_SERVER_NAME, parse_usage, split_last_user, strip_mcp_prefix, write_fake_session,
};

// ── Stub tools ───────────────────────────────────────────────────────────────

/// Surfaces caller-provided tool schemas to claude without executing anything.
///
/// We kill the subprocess on the first `assistant` message, but that kill is
/// asynchronous — claude may still hit the MCP server before SIGKILL lands.
/// Returning an empty result instantly prevents a blocked tool-call response
/// from pinning the subprocess alive until our drop cleanup.
struct StubTools {
    defs: Vec<ToolDefinition>,
}

#[async_trait]
impl Tool for StubTools {
    fn raw_tools(&self) -> Vec<ToolDefinition> {
        self.defs.clone()
    }
    async fn call(&self, _name: &str, _args: serde_json::Value) -> BoxStream<'static, ToolOutput> {
        futures::stream::iter(vec![ToolOutput::Result(vec![])]).boxed()
    }
}

// ── Subprocess setup (shared) ────────────────────────────────────────────────

/// Build the MCP server, write the config + fake session files, spawn
/// `claude -p`, and feed the user message on stdin. Returns the Cleanup guard
/// and the live Child — caller drives stdout and is responsible for both
/// `drop(child)` (SIGKILL via `kill_on_drop`) and `drop(guard)` (abort MCP
/// task + remove temp files), **in that order**.
async fn start_claude(
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<(Cleanup, Child), ApiError> {
    let (prev_history, mut last_user_content) =
        split_last_user(messages.to_vec()).map_err(ApiError::Other)?;

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let mcp_addr = listener.local_addr()?;
    let stub = StubTools {
        defs: tools.to_vec(),
    };
    let router = McpServer::new(stub).into_axum_router();
    let mcp_task = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });

    let mut guard = Cleanup::new(mcp_task);

    let mcp_config_path =
        std::env::temp_dir().join(format!("agentix-mcp-{}.json", uuid::Uuid::new_v4()));
    let mcp_config = serde_json::json!({
        "mcpServers": {
            MCP_SERVER_NAME: {
                "type": "http",
                "url": format!("http://{}/", mcp_addr),
            }
        }
    });
    tokio::fs::write(&mcp_config_path, mcp_config.to_string())
        .await
        .map_err(|e| ApiError::Other(format!("write mcp-config: {e}")))?;
    guard.temp_files.push(mcp_config_path.clone());

    let mut resume_args: Vec<String> = Vec::new();
    if !prev_history.is_empty() {
        let (sid, path, id_map) = write_fake_session(&prev_history)
            .await
            .map_err(|e| ApiError::Other(format!("write fake session: {e}")))?;
        guard.temp_files.push(path);
        resume_args.push("--resume".into());
        resume_args.push(sid);
        // Rewrite any tool_use_ids in the stdin message to match the remapped
        // ids in the resumed session.
        self::session::remap_tool_use_ids(&mut last_user_content, &id_map);
    }

    let mut args: Vec<String> = vec![
        "-p".into(),
        "--strict-mcp-config".into(),
        "--mcp-config".into(),
        mcp_config_path.to_string_lossy().into_owned(),
        "--tools".into(),
        String::new(),
        "--output-format".into(),
        "stream-json".into(),
        "--input-format".into(),
        "stream-json".into(),
        "--include-partial-messages".into(),
        "--verbose".into(),
        "--permission-mode".into(),
        "bypassPermissions".into(),
        "--no-session-persistence".into(),
    ];
    if let Some(sp) = &config.system_prompt {
        args.push("--system-prompt".into());
        args.push(sp.clone());
    }
    if !config.model.is_empty() {
        args.push("--model".into());
        args.push(config.model.clone());
    }
    args.extend(resume_args);

    info!(args_len = args.len(), "spawning claude-code");
    debug!(?args, "claude-code argv");

    let mut cmd = Command::new("claude");
    cmd.args(&args)
        .env("IS_SANDBOX", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = cmd
        .spawn()
        .map_err(|e| ApiError::Other(format!("spawn claude: {e}")))?;

    if let Some(mut stdin) = child.stdin.take() {
        let msg = serde_json::json!({
            "type": "user",
            "message": {
                "role": "user",
                "content": last_user_content,
            }
        });
        let mut line = msg.to_string();
        line.push('\n');
        if let Err(e) = stdin.write_all(line.as_bytes()).await {
            warn!(error = %e, "write stdin");
        }
        drop(stdin);
    }

    if let Some(err) = child.stderr.take() {
        tokio::spawn(async move {
            let mut elines = BufReader::new(err).lines();
            while let Ok(Some(l)) = elines.next_line().await {
                warn!(target: "claude_code_stderr", "{}", l);
            }
        });
    }

    Ok((guard, child))
}

// ── Stream-JSON → LlmEvent (partial deltas) ─────────────────────────────────

#[derive(Default)]
struct StreamState {
    tool_bufs: Vec<Option<PartialToolCall>>,
}

fn translate_stream_event_line(v: &serde_json::Value, state: &mut StreamState) -> Vec<LlmEvent> {
    let ty = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
    if ty != "stream_event" {
        return Vec::new();
    }
    let ev = match v.get("event") {
        Some(e) => e,
        None => return Vec::new(),
    };
    let ety = ev.get("type").and_then(|x| x.as_str()).unwrap_or("");
    let mut out = Vec::new();

    match ety {
        "content_block_start" => {
            let idx = ev.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let block = match ev.get("content_block") {
                Some(b) => b,
                None => return out,
            };
            if block.get("type").and_then(|x| x.as_str()) == Some("tool_use") {
                let id = block
                    .get("id")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let raw_name = block
                    .get("name")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = strip_mcp_prefix(&raw_name);
                if state.tool_bufs.len() <= idx {
                    state.tool_bufs.resize_with(idx + 1, || None);
                }
                state.tool_bufs[idx] = Some(PartialToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: String::new(),
                });
                out.push(LlmEvent::ToolCallChunk(ToolCallChunk {
                    id,
                    name,
                    delta: String::new(),
                    index: idx as u32,
                }));
            }
        }
        "content_block_delta" => {
            let idx = ev.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let delta = match ev.get("delta") {
                Some(d) => d,
                None => return out,
            };
            match delta.get("type").and_then(|x| x.as_str()).unwrap_or("") {
                "text_delta" => {
                    if let Some(t) = delta.get("text").and_then(|x| x.as_str())
                        && !t.is_empty()
                    {
                        out.push(LlmEvent::Token(t.to_string()));
                    }
                }
                "thinking_delta" => {
                    if let Some(t) = delta.get("thinking").and_then(|x| x.as_str())
                        && !t.is_empty()
                    {
                        out.push(LlmEvent::Reasoning(t.to_string()));
                    }
                }
                "input_json_delta" => {
                    if let Some(partial_json) = delta.get("partial_json").and_then(|x| x.as_str())
                        && !partial_json.is_empty()
                        && let Some(Some(partial)) = state.tool_bufs.get_mut(idx)
                    {
                        partial.arguments.push_str(partial_json);
                        out.push(LlmEvent::ToolCallChunk(ToolCallChunk {
                            id: partial.id.clone(),
                            name: partial.name.clone(),
                            delta: partial_json.to_string(),
                            index: idx as u32,
                        }));
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
    out
}

// ── stream_claude_code ──────────────────────────────────────────────────────

pub(crate) async fn stream_claude_code(
    _token: &str,
    _http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
    let (guard, mut child) = start_claude(config, messages, tools).await?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ApiError::Other("claude subprocess has no stdout".into()))?;
    let mut lines = BufReader::new(stdout).lines();

    Ok(stream! {
        // Moved into the generator; explicit drops below order cleanup.
        let guard = guard;
        let mut child = child;
        let mut state = StreamState::default();
        let mut got_terminal = false;

        'outer: loop {
            let line = match lines.next_line().await {
                Ok(Some(l)) => l,
                Ok(None) => break,
                Err(e) => {
                    yield LlmEvent::Error(format!("read stdout: {e}"));
                    got_terminal = true;
                    break;
                }
            };
            if line.trim().is_empty() { continue; }
            let v: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    warn!(error = %e, line = %line, "malformed stream-json line");
                    continue;
                }
            };

            for ev in translate_stream_event_line(&v, &mut state) {
                yield ev;
            }

            let ty = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
            match ty {
                "assistant" => {
                    // A single turn can produce multiple `assistant` events when
                    // extended thinking is active: a thinking-only message first,
                    // then the "real" message with text/tool_use. Treat the turn
                    // as finished only when we see a non-thinking block.
                    let msg = match v.get("message") { Some(m) => m, None => continue };
                    let content = msg.get("content").and_then(|c| c.as_array());
                    let has_payload = content
                        .map(|arr| arr.iter().any(|b| {
                            matches!(
                                b.get("type").and_then(|x| x.as_str()),
                                Some("text") | Some("tool_use")
                            )
                        }))
                        .unwrap_or(false);

                    if let Some(u) = msg.get("usage") {
                        yield LlmEvent::Usage(parse_usage(u));
                    }

                    if !has_payload {
                        // Thinking-only assistant; wait for the next one.
                        continue;
                    }

                    if let Some(blocks) = content {
                        for block in blocks {
                            if block.get("type").and_then(|x| x.as_str()) == Some("tool_use") {
                                let id = block.get("id").and_then(|x| x.as_str()).unwrap_or("").to_string();
                                let raw_name = block.get("name").and_then(|x| x.as_str()).unwrap_or("");
                                let input = block.get("input").cloned().unwrap_or(serde_json::json!({}));
                                let arguments = serde_json::to_string(&input).unwrap_or_default();
                                yield LlmEvent::ToolCall(ToolCall {
                                    id,
                                    name: strip_mcp_prefix(raw_name),
                                    arguments,
                                });
                            }
                        }
                    }
                    yield LlmEvent::Done;
                    got_terminal = true;
                    break 'outer;
                }
                "result" => {
                    let subtype = v.get("subtype").and_then(|x| x.as_str()).unwrap_or("");
                    let is_error = v.get("is_error").and_then(|x| x.as_bool()).unwrap_or(false);
                    if subtype == "success" && !is_error {
                        yield LlmEvent::Done;
                    } else {
                        warn!(payload = %v, "claude-code non-success result");
                        let msg = v.get("result")
                            .and_then(|x| x.as_str())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| {
                                if subtype.is_empty() {
                                    "unknown error".to_string()
                                } else {
                                    subtype.to_string()
                                }
                            });
                        yield LlmEvent::Error(msg);
                    }
                    got_terminal = true;
                    break 'outer;
                }
                _ => {}
            }
        }

        if !got_terminal {
            match child.wait().await {
                Ok(status) if status.success() => {
                    yield LlmEvent::Error(
                        "claude exited without emitting assistant or result".into(),
                    );
                }
                Ok(status) => {
                    yield LlmEvent::Error(format!("claude exited with status {status}"));
                }
                Err(e) => {
                    yield LlmEvent::Error(format!("wait claude: {e}"));
                }
            }
        }

        drop(child);
        drop(guard);
    }
    .boxed())
}

// ── complete_claude_code ────────────────────────────────────────────────────

pub(crate) async fn complete_claude_code(
    _token: &str,
    _http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<CompleteResponse, ApiError> {
    let (guard, mut child) = start_claude(config, messages, tools).await?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ApiError::Other("claude subprocess has no stdout".into()))?;
    let mut lines = BufReader::new(stdout).lines();

    let mut content_buf = String::new();
    let mut reasoning_buf = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut usage = UsageStats::default();
    // stream-json input mode never populates `message.stop_reason` or emits a
    // `message_delta` event — the only signal is whether the assistant turn
    // produced tool_use blocks. Overridden below if the CLI ever does send a
    // concrete stop_reason.
    let mut finish_reason: Option<FinishReason> = None;
    let mut err: Option<ApiError> = None;
    let mut got_terminal = false;

    loop {
        let line = match lines.next_line().await {
            Ok(Some(l)) => l,
            Ok(None) => break,
            Err(e) => {
                err = Some(ApiError::Stream(format!("read stdout: {e}")));
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, line = %line, "malformed stream-json line");
                continue;
            }
        };
        match v.get("type").and_then(|x| x.as_str()).unwrap_or("") {
            "assistant" => {
                // Extended thinking produces a thinking-only `assistant` event
                // before the real payload. Don't terminate on it.
                let msg = match v.get("message") {
                    Some(m) => m,
                    None => continue,
                };
                let mut saw_payload = false;
                if let Some(content) = msg.get("content").and_then(|c| c.as_array()) {
                    for block in content {
                        match block.get("type").and_then(|x| x.as_str()).unwrap_or("") {
                            "text" => {
                                if let Some(t) = block.get("text").and_then(|x| x.as_str()) {
                                    content_buf.push_str(t);
                                    saw_payload = true;
                                }
                            }
                            "thinking" => {
                                if let Some(t) = block.get("thinking").and_then(|x| x.as_str()) {
                                    reasoning_buf.push_str(t);
                                }
                            }
                            "tool_use" => {
                                let id = block
                                    .get("id")
                                    .and_then(|x| x.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let raw_name =
                                    block.get("name").and_then(|x| x.as_str()).unwrap_or("");
                                let input =
                                    block.get("input").cloned().unwrap_or(serde_json::json!({}));
                                let arguments = serde_json::to_string(&input).unwrap_or_default();
                                tool_calls.push(ToolCall {
                                    id,
                                    name: strip_mcp_prefix(raw_name),
                                    arguments,
                                });
                                saw_payload = true;
                            }
                            _ => {}
                        }
                    }
                }
                if let Some(u) = msg.get("usage") {
                    usage = parse_usage(u);
                }
                if let Some(sr) = msg.get("stop_reason").and_then(|x| x.as_str()) {
                    finish_reason = Some(FinishReason::from(sr));
                }
                if !saw_payload {
                    continue;
                }
                got_terminal = true;
                break;
            }
            "result" => {
                let subtype = v.get("subtype").and_then(|x| x.as_str()).unwrap_or("");
                let is_error = v.get("is_error").and_then(|x| x.as_bool()).unwrap_or(false);
                if subtype != "success" || is_error {
                    warn!(payload = %v, "claude-code non-success result");
                    let msg = v
                        .get("result")
                        .and_then(|x| x.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| {
                            if subtype.is_empty() {
                                "unknown error".to_string()
                            } else {
                                subtype.to_string()
                            }
                        });
                    err = Some(ApiError::Llm(msg));
                }
                got_terminal = true;
                break;
            }
            _ => {}
        }
    }

    if err.is_none() && !got_terminal {
        err = Some(match child.wait().await {
            Ok(status) if status.success() => {
                ApiError::Other("claude exited without emitting assistant or result".into())
            }
            Ok(status) => ApiError::Other(format!("claude exited with status {status}")),
            Err(e) => ApiError::Other(format!("wait claude: {e}")),
        });
    }

    drop(child);
    drop(guard);

    if let Some(e) = err {
        return Err(e);
    }

    let finish_reason = finish_reason.unwrap_or({
        if tool_calls.is_empty() {
            FinishReason::Stop
        } else {
            FinishReason::ToolCalls
        }
    });

    Ok(CompleteResponse {
        content: if content_buf.is_empty() {
            None
        } else {
            Some(content_buf)
        },
        reasoning: if reasoning_buf.is_empty() {
            None
        } else {
            Some(reasoning_buf)
        },
        tool_calls,
        usage,
        finish_reason,
    })
}
