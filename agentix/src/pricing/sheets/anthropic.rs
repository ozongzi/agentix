//! Anthropic first-party price sheets (USD per 1M tokens).
//!
//! Snapshot of <https://platform.claude.com/docs/en/about-claude/pricing>,
//! 2026-10. Billing rules: `input_tokens` excludes cache tokens; cache write =
//! 1.25× input (5m TTL) / 2× (1h TTL) across the lineup, while cache **read**
//! is 0.1× input except for two documented exceptions — 0.025× on Fable 5.1 /
//! Mythos 5.1 and 0.05× on Opus 5.5. Thinking bills as output. No long-context
//! surcharge: Claude 4.6+ include the full 1M window at standard rates. Batch
//! = 50%; US-only inference adds 1.1×; Fast mode on Opus 5.5 is $8 / $40.
//!
//! Current lineup: Fable 5.1 ($10 / $50), Opus 5.5 ($4 / $20),
//! Sonnet 5.5 ($2 / $10), Haiku 4.5 ($1 / $5). Both Opus and Sonnet got
//! *cheaper* at the 5 generation (Opus 4.6–4.8 were $5 / $25; Sonnet 4.x was
//! $3 / $15), so the match is generation-aware rather than family-wide. IDs
//! are dateless from the 4.6 generation on (`claude-opus-5-5`), so no date
//! parsing is needed.
//!
//! Also used by the Claude Code provider, whose `opus` / `sonnet` / `fable`
//! aliases carry no version and resolve to the newest release of their family
//! — so a bare alias is priced at the current generation. Under a Max/Pro
//! subscription there is no marginal cost and the estimate is informational.

use crate::pricing::{PriceSheet, Rates, dec};
use rust_decimal::Decimal;
use rusty_money::iso;

/// input / output per 1M USD. Writes use Anthropic's standard multipliers
/// (1.25× for 5m, 2× for 1h); the cache-read multiplier is explicit because
/// Fable 5.1 and Opus 5.5 are documented exceptions to the 0.1× rule.
fn anthropic_rates(input: &str, output: &str, cache_read_mult: &str) -> Rates {
    let input = dec(input);
    Rates {
        input,
        cache_read: input * dec(cache_read_mult),
        cache_write_5m: input * dec("1.25"),
        cache_write_1h: Some(input * Decimal::TWO),
        output: dec(output),
        ..Default::default()
    }
}

pub fn sheet(model: &str) -> Option<PriceSheet> {
    let m = model.to_ascii_lowercase();
    let rates = if m == "fable" {
        // Claude Code aliases carry no version and resolve to the newest
        // release of their family — price them at the current generation.
        anthropic_rates("10", "50", "0.025")
    } else if m == "opus" {
        anthropic_rates("4", "20", "0.05")
    } else if m == "sonnet" {
        anthropic_rates("2", "10", "0.1")
    } else if m.contains("fable-5") || m.contains("mythos-5") {
        anthropic_rates("10", "50", "0.025")
    } else if m.contains("opus-5") {
        anthropic_rates("4", "20", "0.05")
    } else if m.contains("sonnet-5") {
        anthropic_rates("2", "10", "0.1")
    } else if m.contains("opus") {
        // Opus 4.6 / 4.7 / 4.8 share $5 / $25.
        anthropic_rates("5", "25", "0.1")
    } else if m.contains("sonnet") {
        // Sonnet 4.x standard price.
        anthropic_rates("3", "15", "0.1")
    } else if m.contains("haiku") {
        anthropic_rates("1", "5", "0.1")
    } else {
        return None;
    };
    Some(PriceSheet::flat(iso::USD, rates))
}
