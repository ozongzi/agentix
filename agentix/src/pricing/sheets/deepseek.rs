//! DeepSeek price sheets (USD per 1M tokens).
//!
//! Snapshot of <https://api-docs.deepseek.com/quick_start/pricing>, 2026-10.
//! Billing rules: `prompt_tokens = cache_hit + cache_miss` — a split, priced
//! per side (input bucket = miss price, cache_read bucket = hit price).
//! Automatic disk caching, no write/storage fee. Reasoning billed as output.
//!
//! DeepSeek bills peak and off-peak rates — peak is 01:00–04:00 and
//! 06:00–10:00 UTC, Mon–Fri; off-peak is exactly half. A static sheet cannot
//! see when a request ran, so this models the **peak** (higher) rate, the
//! conservative choice for budget enforcement; halve every figure for
//! off-peak. (The 2026-07 revision of this file wrongly stated the off-peak
//! program had ended.)
//!
//! The current lineup is `deepseek-flash` (DeepSeek-V4.1-Flash, 1M ctx,
//! vision) and `deepseek-v4-pro` (DeepSeek-V4-Pro-0813, no vision). The
//! retired names `deepseek-v4-flash` / `deepseek-v4-flash-vision-exp` are
//! still accepted and served by V4.1-Flash at the Flash price;
//! `deepseek-chat` / `-reasoner` are deprecated legacy names for the Pro
//! line (deprecated 2026-07-24).

use crate::pricing::{PriceSheet, Rates, dec};
use rusty_money::iso;

pub fn sheet(model: &str) -> Option<PriceSheet> {
    let m = model.to_ascii_lowercase();
    let rates = if m.contains("flash") {
        Rates::simple(dec("0.30"), dec("0.006"), dec("1.20"))
    } else if m.contains("pro") || m.contains("chat") || m.contains("reasoner") {
        Rates::simple(dec("1.32"), dec("0.044"), dec("3.96"))
    } else {
        return None;
    };
    Some(PriceSheet::flat(iso::USD, rates))
}
