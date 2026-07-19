//! DeepSeek price sheets (USD per 1M tokens).
//!
//! Snapshot of <https://api-docs.deepseek.com/quick_start/pricing>, 2026-07.
//! Billing rules: `prompt_tokens = cache_hit + cache_miss` — a split, priced
//! per side (input bucket = miss price, cache_read bucket = hit price).
//! Automatic disk caching, no write/storage fee. Reasoning billed as output.
//! The 2025 off-peak discount program has ended — flat pricing.

use crate::pricing::{PriceSheet, Rates, dec};
use rusty_money::iso;

pub fn sheet(model: &str) -> Option<PriceSheet> {
    let m = model.to_ascii_lowercase();
    let rates = if m.contains("flash") {
        Rates::simple(dec("0.14"), dec("0.0028"), dec("0.28"))
    } else if m.contains("pro") || m.contains("chat") || m.contains("reasoner") {
        // deepseek-v4-pro; legacy deepseek-chat/-reasoner route here too
        // (deprecated 2026-07-24).
        Rates::simple(dec("0.435"), dec("0.003625"), dec("0.87"))
    } else {
        return None;
    };
    Some(PriceSheet::flat(iso::USD, rates))
}
