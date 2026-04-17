//! Claude Code agent — drives `claude -p` as the agentic loop, exposing
//! agentix tools via an in-process MCP server.
//!
//! Enable with the `claude-code` feature flag (requires `mcp-server`):
//!
//! ```toml
//! agentix = { version = "0.16", features = ["claude-code"] }
//! ```
//!
//! # Why
//!
//! Ride a Claude Max OAuth subscription (read from keychain by the `claude`
//! CLI) instead of paying per-token via `ANTHROPIC_API_KEY`. The agentic
//! loop runs inside `claude -p`; tool calls dispatch back through a loopback
//! MCP server that serves [`agentix::Tool`]s.
//!
//! # Example
//!
//! ```no_run
//! # use agentix::{agent_claude_code, ClaudeCodeConfig, Message, UserContent, AgentEvent};
//! # use futures::StreamExt;
//! # async fn run() {
//! let history = vec![Message::User(vec![UserContent::text("What is 2+2?")])];
//!
//! let mut stream = agent_claude_code(
//!     agentix::ToolBundle::default(),
//!     "You are a concise math helper.",
//!     ClaudeCodeConfig::new().model("sonnet"),
//!     history,
//! );
//!
//! while let Some(event) = stream.next().await {
//!     if let AgentEvent::Token(t) = event { print!("{t}"); }
//! }
//! # }
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;

use async_stream::stream;
use futures::stream::BoxStream;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::process::Command;
use tracing::{debug, info, warn};

use crate::agent::AgentEvent;
use crate::mcp_server::McpServer;
use crate::request::{Content, Message, ToolCall};
use crate::tool_trait::Tool;
use crate::types::UsageStats;

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for [`agent_claude_code`].
///
/// All fields are optional — use [`ClaudeCodeConfig::new`] plus builder
/// methods to customize, or rely on `Default`.
#[derive(Default, Clone, Debug)]
pub struct ClaudeCodeConfig {
    /// `--model` flag value (e.g. `"sonnet"`, `"opus"`, `"haiku"`).
    /// When `None`, `claude` picks its default.
    pub model: Option<String>,

    /// Path to the `claude` binary. When `None`, spawns `"claude"` and
    /// relies on `$PATH` resolution.
    pub claude_binary: Option<PathBuf>,

    /// Extra CLI arguments appended *before* our fixed flags — our flags
    /// take precedence under `claude`'s last-wins parsing for duplicates.
    pub extra_args: Vec<String>,
}

impl ClaudeCodeConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = Some(m.into());
        self
    }

    pub fn binary(mut self, p: impl Into<PathBuf>) -> Self {
        self.claude_binary = Some(p.into());
        self
    }

    pub fn arg(mut self, a: impl Into<String>) -> Self {
        self.extra_args.push(a.into());
        self
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Drive `claude -p` as the agentic loop, exposing `tools` via an in-process
/// MCP server, and yield [`AgentEvent`]s compatible with [`crate::agent`].
///
/// - `history` — full conversation so far. The last element **must** be a
///   `Message::User`; everything before it is replayed via `--resume` as
///   pre-existing session state.
/// - `system_prompt` — replaces Claude Code's default system prompt.
///
/// Dropping the returned stream aborts the subprocess and cleans up the
/// temporary MCP-config file and fake session file.
pub fn agent_claude_code(
    tools: impl Tool + 'static,
    system_prompt: impl Into<String>,
    config: ClaudeCodeConfig,
    history: Vec<Message>,
) -> BoxStream<'static, AgentEvent> {
    let system_prompt = system_prompt.into();

    Box::pin(stream! {
        // ── Validate history ─────────────────────────────────────────────
        let (prev_history, last_user_content) = match split_last_user(history) {
            Ok(v) => v,
            Err(e) => { yield AgentEvent::Error(e); return; }
        };

        // ── Start MCP server on a random loopback port ───────────────────
        let listener = match TcpListener::bind("127.0.0.1:0").await {
            Ok(l) => l,
            Err(e) => { yield AgentEvent::Error(format!("bind MCP server: {e}")); return; }
        };
        let mcp_addr = match listener.local_addr() {
            Ok(a) => a,
            Err(e) => { yield AgentEvent::Error(format!("local_addr: {e}")); return; }
        };
        let router = McpServer::new(tools).into_axum_router();
        let mcp_task = tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });

        // Guard cleans up temp files + aborts subprocess/mcp on drop.
        let mut guard = Cleanup::new(mcp_task);

        // ── Write mcp-config JSON to a temp file ─────────────────────────
        let mcp_config_path = std::env::temp_dir()
            .join(format!("agentix-mcp-{}.json", uuid::Uuid::new_v4()));
        let mcp_config = serde_json::json!({
            "mcpServers": {
                MCP_SERVER_NAME: {
                    "type": "http",
                    "url": format!("http://{}/", mcp_addr),
                }
            }
        });
        if let Err(e) = tokio::fs::write(&mcp_config_path, mcp_config.to_string()).await {
            yield AgentEvent::Error(format!("write mcp-config: {e}"));
            return;
        }
        guard.temp_files.push(mcp_config_path.clone());

        // ── Fabricate session file (when there's prior history) ──────────
        let mut resume_args: Vec<String> = Vec::new();
        if !prev_history.is_empty() {
            match write_fake_session(&prev_history).await {
                Ok((sid, path)) => {
                    guard.temp_files.push(path);
                    resume_args.push("--resume".into());
                    resume_args.push(sid);
                }
                Err(e) => {
                    yield AgentEvent::Error(format!("write fake session: {e}"));
                    return;
                }
            }
        }

        // ── Build argv ───────────────────────────────────────────────────
        let binary = config.claude_binary
            .clone()
            .unwrap_or_else(|| PathBuf::from("claude"));

        let mut args: Vec<String> = Vec::new();
        // User extras first — so our fixed flags (appended after) win under
        // Commander.js's last-wins semantics for duplicated options.
        args.extend(config.extra_args.iter().cloned());
        args.push("-p".into());
        args.push("--strict-mcp-config".into());
        args.push("--mcp-config".into());
        args.push(mcp_config_path.to_string_lossy().into_owned());
        args.push("--tools".into());
        args.push(String::new()); // empty string disables all built-in tools
        args.push("--output-format".into());
        args.push("stream-json".into());
        args.push("--input-format".into());
        args.push("stream-json".into());
        args.push("--include-partial-messages".into());
        args.push("--verbose".into()); // required by stream-json output
        args.push("--permission-mode".into());
        args.push("bypassPermissions".into());
        args.push("--no-session-persistence".into());
        args.push("--system-prompt".into());
        args.push(system_prompt);
        if let Some(m) = &config.model {
            args.push("--model".into());
            args.push(m.clone());
        }
        args.extend(resume_args);

        info!(binary = %binary.display(), args_len = args.len(), "spawning claude-code");
        debug!(?args, "claude-code argv");

        // ── Spawn subprocess ─────────────────────────────────────────────
        // IS_SANDBOX=1 lets claude accept --dangerously-skip-permissions when
        // running as root (typical inside Docker).
        let mut cmd = Command::new(&binary);
        cmd.args(&args)
            .env("IS_SANDBOX", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = match cmd.spawn()
        {
            Ok(c) => c,
            Err(e) => {
                yield AgentEvent::Error(format!("spawn {}: {e}", binary.display()));
                return;
            }
        };

        // ── Feed the current user message via stdin (stream-json) ────────
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
            // Close stdin so claude knows no more user input is coming.
            drop(stdin);
        }

        let stderr = child.stderr.take();
        let stdout = match child.stdout.take() {
            Some(s) => s,
            None => {
                yield AgentEvent::Error("claude subprocess has no stdout".into());
                return;
            }
        };
        let mut lines = BufReader::new(stdout).lines();

        // Drain stderr into a tracing channel so it doesn't fill up.
        // Log at warn so default `info` filters still surface claude CLI
        // diagnostics — silent MCP/auth/resume failures have been hard to
        // diagnose otherwise.
        if let Some(err) = stderr {
            tokio::spawn(async move {
                let mut elines = BufReader::new(err).lines();
                while let Ok(Some(l)) = elines.next_line().await {
                    warn!(target: "claude_code_stderr", "{}", l);
                }
            });
        }

        // ── Parse stream-json lines → AgentEvents ────────────────────────
        let mut tool_ids_to_names: HashMap<String, String> = HashMap::new();
        let mut total_usage = UsageStats::default();
        let mut done = false;

        'outer: loop {
            let line = match lines.next_line().await {
                Ok(Some(l)) => l,
                Ok(None) => break,
                Err(e) => {
                    yield AgentEvent::Error(format!("read stdout: {e}"));
                    done = true;
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
            for event in translate_stream_json(&v, &mut tool_ids_to_names, &mut total_usage) {
                let terminal = matches!(event, AgentEvent::Done(_) | AgentEvent::Error(_));
                yield event;
                if terminal {
                    done = true;
                    break 'outer;
                }
            }
        }

        // ── Wait for process exit to surface non-stream-json errors ──────
        // Skip if we already yielded a terminal event — otherwise consumers
        // would see a Done followed by a spurious Error from a non-zero exit.
        if !done {
            match child.wait().await {
                Ok(status) if !status.success() => {
                    yield AgentEvent::Error(format!("claude exited with status {status}"));
                }
                Err(e) => {
                    yield AgentEvent::Error(format!("wait claude: {e}"));
                }
                _ => {}
            }
        }

        drop(guard);
    })
}

// ── Cleanup guard ─────────────────────────────────────────────────────────────

struct Cleanup {
    mcp_task: Option<tokio::task::JoinHandle<()>>,
    temp_files: Vec<PathBuf>,
}

impl Cleanup {
    fn new(mcp_task: tokio::task::JoinHandle<()>) -> Self {
        Self {
            mcp_task: Some(mcp_task),
            temp_files: Vec::new(),
        }
    }
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        if let Some(t) = self.mcp_task.take() {
            t.abort();
        }
        for path in std::mem::take(&mut self.temp_files) {
            let _ = std::fs::remove_file(&path);
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const MCP_SERVER_NAME: &str = "agentix";

/// Split off the last User message; the return value is `(prev_history,
/// stream_json_content)` where `stream_json_content` is the JSON shape Claude
/// Code expects for a stdin user message.
fn split_last_user(
    history: Vec<Message>,
) -> Result<(Vec<Message>, serde_json::Value), String> {
    if history.is_empty() {
        return Err("history is empty; need at least one User message".into());
    }
    let mut history = history;
    let last = history.pop().expect("non-empty");
    let content = match last {
        Message::User(parts) => user_content_to_json(&parts),
        _ => return Err("last message in history must be Message::User".into()),
    };
    Ok((history, content))
}

fn user_content_to_json(parts: &[Content]) -> serde_json::Value {
    // Simple case: single text part → plain string. Otherwise an array of
    // Anthropic-shaped content blocks.
    if let [Content::Text { text }] = parts {
        return serde_json::Value::String(text.clone());
    }
    let blocks: Vec<serde_json::Value> = parts
        .iter()
        .map(|p| match p {
            Content::Text { text } => serde_json::json!({
                "type": "text",
                "text": text,
            }),
            Content::Image(img) => {
                let (src_type, src_field, src_value) = match &img.data {
                    crate::request::ImageData::Base64(b) => ("base64", "data", b.clone()),
                    crate::request::ImageData::Url(u) => ("url", "url", u.clone()),
                };
                serde_json::json!({
                    "type": "image",
                    "source": {
                        "type": src_type,
                        "media_type": img.mime_type,
                        src_field: src_value,
                    }
                })
            }
        })
        .collect();
    serde_json::Value::Array(blocks)
}

/// Convert `~/.claude/projects/<sanitized_cwd>/<uuid>.jsonl` via claude's
/// scheme: replace every non-alphanumeric byte with `-`, truncate + hash
/// when longer than 200 bytes.
fn sanitize_cwd(cwd: &std::path::Path) -> String {
    let s = cwd.to_string_lossy();
    let sanitized: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    const MAX: usize = 200;
    if sanitized.len() <= MAX {
        sanitized
    } else {
        // Trail with a short hash to preserve uniqueness. Use std's hasher.
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut h);
        format!("{}-{:x}", &sanitized[..MAX], h.finish())
    }
}

/// Write a fake session jsonl to `~/.claude/projects/<sanitized_cwd>/<sid>.jsonl`
/// containing the given history, and return (session_id, path).
///
/// Tool-call ids from other providers are rewritten to Anthropic-shaped
/// `toolu_<hex>` so claude accepts them and so tool_use/tool_result pairs
/// stay matched after rewriting.
async fn write_fake_session(
    history: &[Message],
) -> Result<(String, PathBuf), String> {
    let claude_home = std::env::var_os("CLAUDE_CONFIG_DIR")
        .map(PathBuf::from)
        .or_else(|| dirs_home().map(|h| h.join(".claude")))
        .ok_or("cannot resolve ~/.claude directory")?;

    let cwd = std::env::current_dir().map_err(|e| format!("cwd: {e}"))?;
    let proj_dir = claude_home
        .join("projects")
        .join(sanitize_cwd(&cwd));
    tokio::fs::create_dir_all(&proj_dir)
        .await
        .map_err(|e| format!("mkdir {}: {e}", proj_dir.display()))?;

    let sid = uuid::Uuid::new_v4().to_string();
    let path = proj_dir.join(format!("{sid}.jsonl"));

    // Rewrite tool_use_ids → toolu_<short-hex> for claude compatibility.
    let mut id_map: HashMap<String, String> = HashMap::new();
    let mut remap = |id: &str| -> String {
        if let Some(new) = id_map.get(id) {
            return new.clone();
        }
        let new = format!("toolu_{}", uuid::Uuid::new_v4().simple());
        id_map.insert(id.to_string(), new.clone());
        new
    };

    let now = chrono_like_now();
    let cwd_str = cwd.to_string_lossy().into_owned();
    let mut parent_uuid: Option<String> = None;
    let mut lines = String::new();

    for msg in history {
        let uuid_ = uuid::Uuid::new_v4().to_string();
        let entry = match msg {
            Message::User(parts) => serde_json::json!({
                "parentUuid": parent_uuid,
                "isSidechain": false,
                "type": "user",
                "message": {
                    "role": "user",
                    "content": user_content_to_json(parts),
                },
                "uuid": uuid_,
                "timestamp": now,
                "sessionId": sid,
                "cwd": cwd_str,
                "userType": "external",
                "entrypoint": "cli",
                "version": env!("CARGO_PKG_VERSION"),
            }),
            Message::Assistant {
                content,
                reasoning: _,
                tool_calls,
            } => {
                let mut blocks = Vec::new();
                if let Some(c) = content {
                    if !c.is_empty() {
                        blocks.push(serde_json::json!({"type": "text", "text": c}));
                    }
                }
                for tc in tool_calls {
                    let new_id = remap(&tc.id);
                    let input: serde_json::Value =
                        serde_json::from_str(&tc.arguments).unwrap_or(serde_json::json!({}));
                    blocks.push(serde_json::json!({
                        "type": "tool_use",
                        "id": new_id,
                        "name": format!("mcp__{}__{}", MCP_SERVER_NAME, tc.name),
                        "input": input,
                    }));
                }
                serde_json::json!({
                    "parentUuid": parent_uuid,
                    "isSidechain": false,
                    "type": "assistant",
                    "message": {
                        "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
                        "type": "message",
                        "role": "assistant",
                        "content": blocks,
                    },
                    "uuid": uuid_,
                    "timestamp": now,
                    "sessionId": sid,
                    "cwd": cwd_str,
                    "userType": "external",
                    "entrypoint": "cli",
                    "version": env!("CARGO_PKG_VERSION"),
                })
            }
            Message::ToolResult { call_id, content } => {
                let new_id = remap(call_id);
                // Concatenate text parts. Images are dropped for now (claude's
                // tool_result expects text or structured content; URL-encoded
                // image parts would need mirroring the mcp_server flow).
                let text = content
                    .iter()
                    .filter_map(|c| {
                        if let Content::Text { text } = c {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                serde_json::json!({
                    "parentUuid": parent_uuid,
                    "isSidechain": false,
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": new_id,
                            "content": text,
                        }]
                    },
                    "uuid": uuid_,
                    "timestamp": now,
                    "sessionId": sid,
                    "cwd": cwd_str,
                    "userType": "external",
                    "entrypoint": "cli",
                    "version": env!("CARGO_PKG_VERSION"),
                })
            }
        };
        lines.push_str(&entry.to_string());
        lines.push('\n');
        parent_uuid = Some(uuid_);
    }

    tokio::fs::write(&path, lines)
        .await
        .map_err(|e| format!("write session {}: {e}", path.display()))?;
    Ok((sid, path))
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn chrono_like_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    // Rough ISO8601 without a date lib — claude only reads it back for display.
    let secs = d.as_secs();
    let ms = d.subsec_millis();
    // Epoch → Y-M-D-H-M-S via /, %, leap-year math would be overkill here.
    // Use a cheap static format; claude parses but doesn't validate semantics.
    let (y, mo, d, h, mi, s) = epoch_to_ymdhms(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{ms:03}Z")
}

fn epoch_to_ymdhms(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let s_in_day = 86_400u64;
    let days = secs / s_in_day;
    let rem = secs % s_in_day;
    let h = (rem / 3600) as u32;
    let mi = ((rem % 3600) / 60) as u32;
    let s = (rem % 60) as u32;

    // Days since 1970-01-01 → Y/M/D using the civil-from-days algorithm
    // (Howard Hinnant, public domain).
    let days = days as i64 + 719_468;
    let era = (if days >= 0 { days } else { days - 146_096 }) / 146_097;
    let doe = (days - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = if m <= 2 { y + 1 } else { y } as u32;
    (y, m, d, h, mi, s)
}

// ── Stream-JSON → AgentEvent translator ──────────────────────────────────────

/// Emits zero or more `AgentEvent`s from a single parsed stream-json line.
fn translate_stream_json(
    v: &serde_json::Value,
    tool_ids_to_names: &mut HashMap<String, String>,
    total_usage: &mut UsageStats,
) -> Vec<AgentEvent> {
    let ty = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
    let mut out = Vec::new();

    match ty {
        // Partial streaming deltas (only when --include-partial-messages).
        "stream_event" => {
            let ev = match v.get("event") {
                Some(e) => e,
                None => return out,
            };
            let ety = ev.get("type").and_then(|x| x.as_str()).unwrap_or("");
            match ety {
                "content_block_delta" => {
                    let delta = match ev.get("delta") {
                        Some(d) => d,
                        None => return out,
                    };
                    let dty = delta.get("type").and_then(|x| x.as_str()).unwrap_or("");
                    match dty {
                        "text_delta" => {
                            if let Some(t) = delta.get("text").and_then(|x| x.as_str()) {
                                out.push(AgentEvent::Token(t.to_string()));
                            }
                        }
                        "thinking_delta" => {
                            if let Some(t) = delta.get("thinking").and_then(|x| x.as_str()) {
                                out.push(AgentEvent::Reasoning(t.to_string()));
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        // Completed assistant message (sent after all stream_events for it).
        "assistant" => {
            let msg = match v.get("message") {
                Some(m) => m,
                None => return out,
            };

            // Extract tool_use blocks → ToolCallStart events.
            if let Some(content) = msg.get("content").and_then(|c| c.as_array()) {
                for block in content {
                    let bty = block.get("type").and_then(|x| x.as_str()).unwrap_or("");
                    if bty == "tool_use" {
                        let id = block
                            .get("id")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = block
                            .get("name")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string();
                        let input = block.get("input").cloned().unwrap_or(serde_json::json!({}));
                        let arguments = serde_json::to_string(&input).unwrap_or_default();

                        // Strip the MCP-server prefix (e.g. `mcp__agentix__foo` → `foo`).
                        let bare_name = strip_mcp_prefix(&name);
                        tool_ids_to_names.insert(id.clone(), bare_name.clone());

                        out.push(AgentEvent::ToolCallStart(ToolCall {
                            id,
                            name: bare_name,
                            arguments,
                        }));
                    }
                }
            }

            // Extract usage.
            if let Some(u) = msg.get("usage") {
                let usage = parse_usage(u);
                *total_usage += usage.clone();
                out.push(AgentEvent::Usage(usage));
            }
        }

        // Tool result produced by claude invoking our MCP server.
        "user" => {
            let msg = match v.get("message") {
                Some(m) => m,
                None => return out,
            };
            let content = match msg.get("content").and_then(|c| c.as_array()) {
                Some(a) => a,
                None => return out,
            };
            for block in content {
                let bty = block.get("type").and_then(|x| x.as_str()).unwrap_or("");
                if bty != "tool_result" { continue; }
                let id = block
                    .get("tool_use_id")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = tool_ids_to_names
                    .get(&id)
                    .cloned()
                    .unwrap_or_default();
                let content = extract_tool_result_content(block);
                out.push(AgentEvent::ToolResult { id, name, content });
            }
        }

        // Final `result` marker — stream is done.
        //
        // Subtype is authoritative: Claude Code CLI sets `subtype:"success"`
        // only on clean completion. We've seen payloads in the wild with
        // `is_error:true` but `subtype:"success"` where the only useful text
        // was literally the word "success" — which produced the baffling
        // user-facing "error: success" message. Trust subtype, and warn-log
        // the raw payload on every non-success so the next real failure in
        // production surfaces its actual shape.
        "result" => {
            let subtype = v.get("subtype").and_then(|x| x.as_str()).unwrap_or("");
            let is_error = v.get("is_error").and_then(|x| x.as_bool()).unwrap_or(false);
            if subtype == "success" && !is_error {
                out.push(AgentEvent::Done(total_usage.clone()));
            } else {
                warn!(payload = %v, "claude-code non-success result");
                let msg = v
                    .get("errors")
                    .and_then(|e| e.as_array())
                    .and_then(|a| a.first())
                    .and_then(|x| {
                        x.as_str()
                            .map(|s| s.to_string())
                            .or_else(|| {
                                x.get("message")
                                    .and_then(|m| m.as_str())
                                    .map(|s| s.to_string())
                            })
                    })
                    .or_else(|| {
                        v.get("result")
                            .and_then(|x| x.as_str())
                            .map(|s| s.to_string())
                    })
                    .unwrap_or_else(|| {
                        if subtype.is_empty() {
                            "unknown error".to_string()
                        } else {
                            subtype.to_string()
                        }
                    });
                out.push(AgentEvent::Error(msg));
            }
        }

        _ => {
            debug!(ty = %ty, payload = %v, "unhandled stream-json type");
        }
    }
    out
}

fn strip_mcp_prefix(name: &str) -> String {
    // Claude Code prefixes MCP tools as `mcp__<server>__<tool>`.
    let pat = format!("mcp__{MCP_SERVER_NAME}__");
    name.strip_prefix(&pat).unwrap_or(name).to_string()
}

fn parse_usage(u: &serde_json::Value) -> UsageStats {
    let get = |k: &str| -> usize {
        u.get(k).and_then(|x| x.as_u64()).unwrap_or(0) as usize
    };
    let prompt = get("input_tokens");
    let completion = get("output_tokens");
    let cache_read = get("cache_read_input_tokens");
    let cache_creation = get("cache_creation_input_tokens");
    UsageStats {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
        cache_read_tokens: cache_read,
        cache_creation_tokens: cache_creation,
    }
}

fn extract_tool_result_content(block: &serde_json::Value) -> Vec<Content> {
    let c = match block.get("content") {
        Some(c) => c,
        None => return Vec::new(),
    };
    match c {
        serde_json::Value::String(s) => vec![Content::text(s)],
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|block| {
                let bty = block.get("type").and_then(|x| x.as_str())?;
                match bty {
                    "text" => block
                        .get("text")
                        .and_then(|x| x.as_str())
                        .map(|s| Content::text(s.to_string())),
                    // Images from tool results are rare in claude output; map
                    // best-effort.
                    "image" => {
                        let source = block.get("source")?;
                        let media_type = source
                            .get("media_type")
                            .and_then(|x| x.as_str())?
                            .to_string();
                        let data = source.get("data").and_then(|x| x.as_str())?;
                        Some(Content::Image(crate::request::ImageContent {
                            mime_type: media_type,
                            data: crate::request::ImageData::Base64(data.to_string()),
                        }))
                    }
                    _ => None,
                }
            })
            .collect(),
        _ => Vec::new(),
    }
}

