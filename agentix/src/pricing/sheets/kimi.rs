//! Moonshot / Kimi price sheets (USD per 1M tokens, international platform).
//!
//! Snapshot of <https://platform.kimi.ai/docs/pricing/chat-k3>, 2026-07.
//! Billing rules: cached tokens are a subset of `prompt_tokens` (hit price
//! vs miss price); caching is automatic with no write or storage fee (the
//! old explicit context-caching API with creation + per-minute storage
//! charges is gone). Output price includes reasoning. Flat 1M context — no
//! length tiers.

use crate::pricing::{PriceSheet, Rates, dec};
use rusty_money::iso;

pub fn sheet(model: &str) -> Option<PriceSheet> {
    let m = model.to_ascii_lowercase();
    if m.contains("k3") || m.contains("kimi") {
        Some(PriceSheet::flat(
            iso::USD,
            Rates::simple(dec("3"), dec("0.30"), dec("15")),
        ))
    } else {
        None
    }
}
