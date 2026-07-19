//! OpenAI price sheets (USD per 1M tokens).
//!
//! Snapshot of <https://developers.openai.com/api/docs/pricing>, 2026-07.
//! Billing rules: `cached_tokens` is a subset of `input_tokens`, billed at
//! 10% of input; reasoning is a subset of output at the output rate; caching
//! is automatic (1024+ token prefixes). The newest (gpt-5.6) family charges
//! cache writes at 1.25× input; earlier families write for free. Batch = 50%.
//!
//! Also used by the Codex provider under API-key auth (ChatGPT-plan auth has
//! no marginal cost).

use crate::pricing::{PriceSheet, Rates, dec};
use rust_decimal::Decimal;
use rusty_money::iso;

fn openai_rates(input: &str, output: &str, paid_cache_write: bool) -> Rates {
    let input = dec(input);
    Rates {
        input,
        cache_read: input * dec("0.1"),
        cache_write_5m: if paid_cache_write {
            input * dec("1.25")
        } else {
            Decimal::ZERO
        },
        output: dec(output),
        ..Default::default()
    }
}

pub fn sheet(model: &str) -> Option<PriceSheet> {
    let m = model.to_ascii_lowercase();
    let write_fee = m.contains("5.6");
    let rates = if m.contains("5.4-nano") {
        openai_rates("0.20", "1.25", write_fee)
    } else if m.contains("5.4-mini") {
        openai_rates("0.75", "4.50", write_fee)
    } else if m.contains("pro") {
        // gpt-5.5-pro / 5.4-pro: no cached rate published.
        Rates {
            input: dec("30"),
            output: dec("180"),
            ..Default::default()
        }
    } else if m.contains("5.6-luna") {
        openai_rates("1", "6", write_fee)
    } else if m.contains("5.6-terra") || m.contains("5.4") {
        openai_rates("2.50", "15", write_fee)
    } else if m.contains("5.6-sol") || m.contains("5.5") {
        openai_rates("5", "30", write_fee)
    } else {
        return None;
    };
    Some(PriceSheet::flat(iso::USD, rates))
}
