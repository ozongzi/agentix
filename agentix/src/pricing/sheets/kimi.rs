//! Moonshot / Kimi price sheets (USD per 1M tokens, international platform).
//!
//! Snapshot of <https://platform.kimi.ai/docs/pricing/chat>, 2026-10.
//! Billing rules: cached input is billed at the cache-hit price only. The K3
//! line additionally bills **cache writes** by TTL tier — $3.00 / 1M for the
//! 5-minute tier (the default when no TTL is given) and $6.00 / 1M for 1h —
//! and a cache hit refreshes the entry's lifetime at no extra write charge.
//! The K2.x line has no write charge. Output price includes reasoning. Flat
//! 1M context — no length tiers.
//!
//! The CN platform (platform.moonshot.cn / platform.kimi.com) quotes the same
//! models in CNY (¥20 / ¥2 / ¥100 for K3); this sheet follows the
//! international USD table.

use crate::pricing::{PriceSheet, Rates, dec};
use rusty_money::iso;

pub fn sheet(model: &str) -> Option<PriceSheet> {
    let m = model.to_ascii_lowercase();
    let rates = if m.contains("k2.7-code-highspeed") {
        Rates::simple(dec("1.90"), dec("0.38"), dec("8"))
    } else if m.contains("k2.7-code") {
        Rates::simple(dec("0.95"), dec("0.19"), dec("4"))
    } else if m.contains("k2.6") {
        Rates::simple(dec("0.95"), dec("0.16"), dec("4"))
    } else if m.contains("k3") || m.contains("kimi") {
        Rates {
            input: dec("3"),
            cache_read: dec("0.30"),
            cache_write_5m: dec("3"),
            cache_write_1h: Some(dec("6")),
            output: dec("15"),
            ..Default::default()
        }
    } else {
        return None;
    };
    Some(PriceSheet::flat(iso::USD, rates))
}
