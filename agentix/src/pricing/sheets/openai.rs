//! OpenAI price sheets (USD per 1M tokens).
//!
//! Snapshot of <https://developers.openai.com/api/docs/pricing>, 2026-10
//! (Standard tier; platform.openai.com 301-redirects to developers.openai.com).
//!
//! Billing rules: cached reads are a *per-model* rate, not one global ratio —
//! 0.05× the uncached input on `gpt-6.1-sol`, 0.1× on most other GPT-5.6+ and
//! later models, and model-specific on older ones (`gpt-5.5` = 0.1×). Cache
//! writes cost 1.25× the uncached input on GPT-5.6 and later and are free
//! before that (`gpt-5.5`). Reasoning bills as output. Every model in the
//! current 6.x / 5.6 / 5.5 families has a distinct **long-context** tier:
//! prompts ≤272K input tokens bill at the short rate, and anything above
//! re-tiers the whole request. Batch is 50% off; Fast (priority) is 2× and
//! Ultrafast 6×; a data-residency/FedRAMP endpoint adds a 10% uplift.
//!
//! Also used by the Codex provider under API-key auth (ChatGPT-plan auth has
//! no marginal cost).

use crate::pricing::{PriceSheet, Rates, Tier, dec};
use rusty_money::iso;

/// Documented short/long context boundary: "Short context: ≤272K input
/// tokens. Long context: >272K input tokens."
const SHORT_CONTEXT_MAX: u64 = 272_000;

/// One context tier. `cache_read` is passed explicitly because OpenAI quotes
/// it per model; `cache_write` is `None` on pre-GPT-5.6 models, which charge
/// no write fee.
fn openai_tier(input: &str, output: &str, cache_read: &str, cache_write: Option<&str>) -> Rates {
    Rates {
        input: dec(input),
        cache_read: dec(cache_read),
        cache_write_5m: cache_write.map(dec).unwrap_or_default(),
        output: dec(output),
        ..Default::default()
    }
}

/// A two-tier sheet: short context, then the re-tiered long-context rates.
fn openai_sheet(short: Rates, long: Rates) -> PriceSheet {
    PriceSheet {
        currency: iso::USD,
        tiers: vec![
            Tier {
                up_to: Some(SHORT_CONTEXT_MAX),
                rates: short,
            },
            Tier {
                up_to: None,
                rates: long,
            },
        ],
    }
}

pub fn sheet(model: &str) -> Option<PriceSheet> {
    let m = model.to_ascii_lowercase();

    // ── Current families, each with a documented long-context tier. ──────
    let tiers = if m.contains("6.1-sol") {
        // Cached read is 0.05× input on this model alone.
        (
            openai_tier("2", "10", "0.10", Some("2.50")),
            openai_tier("4", "15", "0.20", Some("5")),
        )
    } else if m.contains("6-astra") {
        (
            openai_tier("10", "50", "1", Some("12.50")),
            openai_tier("20", "75", "2", Some("25")),
        )
    } else if m.contains("6-sol") {
        (
            openai_tier("2", "10", "0.20", Some("2.50")),
            openai_tier("4", "15", "0.40", Some("5")),
        )
    } else if m.contains("6-luna") {
        (
            openai_tier("0.10", "0.50", "0.01", Some("0.125")),
            openai_tier("0.20", "0.75", "0.02", Some("0.25")),
        )
    } else if m.contains("5.6-sol") {
        (
            openai_tier("4", "20", "0.40", Some("5")),
            openai_tier("8", "30", "0.80", Some("10")),
        )
    } else if m.contains("5.6-terra") {
        (
            openai_tier("2", "12", "0.20", Some("2.50")),
            openai_tier("4", "18", "0.40", Some("5")),
        )
    } else if m.contains("5.6-luna") {
        (
            openai_tier("0.20", "1.20", "0.02", Some("0.25")),
            openai_tier("0.40", "1.80", "0.04", Some("0.50")),
        )
    } else if m.contains("5.5") {
        // Pre-5.6: no cache-write fee.
        (
            openai_tier("5", "30", "0.50", None),
            openai_tier("10", "45", "1", None),
        )
    } else {
        // ── Older families, kept flat: no current long-context data. ──────
        let rates = if m.contains("5.4-nano") {
            openai_tier("0.20", "1.25", "0.02", None)
        } else if m.contains("5.4-mini") {
            openai_tier("0.75", "4.50", "0.075", None)
        } else if m.contains("pro") {
            // gpt-5.5-pro / 5.4-pro: no cached rate published.
            Rates {
                input: dec("30"),
                output: dec("180"),
                ..Default::default()
            }
        } else if m.contains("5.6") || m.contains("5.4") {
            openai_tier("2.50", "15", "0.25", Some("3.125"))
        } else {
            return None;
        };
        return Some(PriceSheet::flat(iso::USD, rates));
    };

    Some(openai_sheet(tiers.0, tiers.1))
}
