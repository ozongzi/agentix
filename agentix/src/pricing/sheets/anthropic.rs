//! Anthropic first-party price sheets (USD per 1M tokens).
//!
//! Snapshot of <https://platform.claude.com/docs/en/pricing>, 2026-07.
//! Billing rules: `input_tokens` excludes cache tokens; cache read = 0.1×
//! input, cache write = 1.25× (5m TTL) / 2× (1h TTL); thinking billed as
//! output; no long-context surcharge on the 1M-window models; Batch = 50%.
//!
//! Also used by the Claude Code provider (its `sonnet` / `opus` aliases) —
//! under a Max/Pro subscription there is no marginal cost and the estimate is
//! informational.

use crate::pricing::{PriceSheet, Rates, dec};
use rust_decimal::Decimal;
use rusty_money::iso;

/// input / output per 1M USD, with Anthropic's standard cache multipliers.
fn anthropic_rates(input: &str, output: &str) -> Rates {
    let input = dec(input);
    Rates {
        input,
        cache_read: input * dec("0.1"),
        cache_write_5m: input * dec("1.25"),
        cache_write_1h: Some(input * Decimal::TWO),
        output: dec(output),
        ..Default::default()
    }
}

pub fn sheet(model: &str) -> Option<PriceSheet> {
    let m = model.to_ascii_lowercase();
    let rates = if m.contains("fable-5") || m.contains("mythos-5") {
        anthropic_rates("10", "50")
    } else if m.contains("opus") {
        // Opus 4.6 / 4.7 / 4.8 share $5 / $25.
        anthropic_rates("5", "25")
    } else if m.contains("sonnet") {
        // Sonnet 4.x / 5 standard price (Sonnet 5 intro pricing not modeled).
        anthropic_rates("3", "15")
    } else if m.contains("haiku") {
        anthropic_rates("1", "5")
    } else {
        return None;
    };
    Some(PriceSheet::flat(iso::USD, rates))
}
