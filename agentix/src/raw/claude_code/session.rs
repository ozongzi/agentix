//! Helpers shared between the `claude_code` raw provider and the top-level
//! `claude_code_agent` driver — session file generation, the subprocess
//! cleanup guard, and stream-json parsing utilities.

use std::collections::HashMap;
use std::path::PathBuf;
use tokio::task::JoinHandle;

use crate::request::{Content, Message};
use crate::types::UsageStats;

/// Name the MCP server is registered under in claude's `--mcp-config`.
/// Tools surface to the model as `mcp__agentix__<tool>`.
pub(crate) const MCP_SERVER_NAME: &str = "agentix";

// ── Cleanup guard ─────────────────────────────────────────────────────────────

/// Aborts the MCP server task and removes temp files on drop.
pub(crate) struct Cleanup {
    mcp_task: Option<JoinHandle<()>>,
    pub(crate) temp_files: Vec<PathBuf>,
}

impl Cleanup {
    pub(crate) fn new(mcp_task: JoinHandle<()>) -> Self {
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

// ── History helpers ───────────────────────────────────────────────────────────

/// Split off the tail of the history that should be fed as the next stdin
/// "user" message to claude, leaving the rest to be replayed via `--resume`.
///
/// Returns `(prev_history, stdin_content)`. `stdin_content` is the JSON
/// "content" shape claude's stream-json stdin expects — a string for a single
/// text, or an array of content blocks otherwise.
///
/// Three shapes are supported:
/// - tail is `Message::User` → text/image blocks, as before.
/// - tail is one or more consecutive `Message::ToolResult`s → a `tool_result`
///   blocks array. Tool-use ids are left in their *original* form here; the
///   caller must remap them using the same id_map returned by
///   [`write_fake_session`] before writing to stdin.
pub(crate) fn split_last_user(
    history: Vec<Message>,
) -> Result<(Vec<Message>, serde_json::Value), String> {
    if history.is_empty() {
        return Err("history is empty; need at least one User/ToolResult message".into());
    }
    let mut history = history;
    match history.last() {
        Some(Message::User(_)) => {
            let Some(Message::User(parts)) = history.pop() else { unreachable!() };
            Ok((history, user_content_to_json(&parts)))
        }
        Some(Message::ToolResult { .. }) => {
            let mut tail: Vec<Message> = Vec::new();
            while matches!(history.last(), Some(Message::ToolResult { .. })) {
                tail.push(history.pop().expect("non-empty"));
            }
            tail.reverse();
            let blocks: Vec<serde_json::Value> = tail
                .into_iter()
                .map(|m| {
                    let Message::ToolResult { call_id, content } = m else { unreachable!() };
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
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": text,
                    })
                })
                .collect();
            Ok((history, serde_json::Value::Array(blocks)))
        }
        _ => Err("last message must be Message::User or Message::ToolResult".into()),
    }
}

/// Rewrite every `tool_use_id` in the given content value using `id_map`. No-op
/// if the value doesn't contain tool_result blocks. Unknown ids pass through
/// unchanged.
pub(crate) fn remap_tool_use_ids(
    content: &mut serde_json::Value,
    id_map: &HashMap<String, String>,
) {
    let arr = match content.as_array_mut() {
        Some(a) => a,
        None => return,
    };
    for block in arr {
        if block.get("type").and_then(|x| x.as_str()) != Some("tool_result") {
            continue;
        }
        let Some(old) = block.get("tool_use_id").and_then(|x| x.as_str()) else {
            continue;
        };
        if let Some(new) = id_map.get(old) {
            if let Some(obj) = block.as_object_mut() {
                obj.insert(
                    "tool_use_id".into(),
                    serde_json::Value::String(new.clone()),
                );
            }
        }
    }
}

pub(crate) fn user_content_to_json(parts: &[Content]) -> serde_json::Value {
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

/// `~/.claude/projects/<sanitized_cwd>/<uuid>.jsonl` — claude's scheme:
/// replace every non-alphanumeric byte with `-`, hash-suffix if > 200 bytes.
pub(crate) fn sanitize_cwd(cwd: &std::path::Path) -> String {
    let s = cwd.to_string_lossy();
    let sanitized: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    const MAX: usize = 200;
    if sanitized.len() <= MAX {
        sanitized
    } else {
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
pub(crate) async fn write_fake_session(
    history: &[Message],
) -> Result<(String, PathBuf, HashMap<String, String>), String> {
    let claude_home = std::env::var_os("CLAUDE_CONFIG_DIR")
        .map(PathBuf::from)
        .or_else(|| dirs_home().map(|h| h.join(".claude")))
        .ok_or("cannot resolve ~/.claude directory")?;

    let cwd = std::env::current_dir().map_err(|e| format!("cwd: {e}"))?;
    let proj_dir = claude_home.join("projects").join(sanitize_cwd(&cwd));
    tokio::fs::create_dir_all(&proj_dir)
        .await
        .map_err(|e| format!("mkdir {}: {e}", proj_dir.display()))?;

    let sid = uuid::Uuid::new_v4().to_string();
    let path = proj_dir.join(format!("{sid}.jsonl"));

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
    Ok((sid, path, id_map))
}

pub(crate) fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

pub(crate) fn chrono_like_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    let ms = d.subsec_millis();
    let (y, mo, d, h, mi, s) = epoch_to_ymdhms(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{ms:03}Z")
}

pub(crate) fn epoch_to_ymdhms(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let s_in_day = 86_400u64;
    let days = secs / s_in_day;
    let rem = secs % s_in_day;
    let h = (rem / 3600) as u32;
    let mi = ((rem % 3600) / 60) as u32;
    let s = (rem % 60) as u32;

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

pub(crate) fn strip_mcp_prefix(name: &str) -> String {
    let pat = format!("mcp__{MCP_SERVER_NAME}__");
    name.strip_prefix(&pat).unwrap_or(name).to_string()
}

pub(crate) fn parse_usage(u: &serde_json::Value) -> UsageStats {
    let get = |k: &str| -> usize { u.get(k).and_then(|x| x.as_u64()).unwrap_or(0) as usize };
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

